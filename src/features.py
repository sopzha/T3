import json
from typing import Optional

import numpy as np

from src.operator_stages import OperatorStage, ExecutionPhase
from src.operators import OperatorType
from src.query_plan import QueryPlan
from src.util import AutoNumber


class Feature(AutoNumber):
    # Features that we have for each operator
    in_card = ()  # cardinality of input used for scan operators (once per pipeline)
    in_size = ()  # size of a tuple scanned by the pipeline (once per pipeline)
    out_card = ()  # cardinality of output used for sink operators (once per pipeline)
    out_size = ()  # size of a tuple materialized at the end of the pipeline (once per pipeline)
    empty_output = ()

    # global features once per pipeline
    pipeline_scan_card = ()
    pipeline_sink_card = ()

    const = ()  # counter for number of each operator type (once per type)
    in_percentage = ()  # percentage of tuples in the pipeline reaching the input of this operator (will be summed up per operator type)
    right_percentage = ()  # percentage of tuples in the pipeline being in the right input of this operator (will be summed up per operator type)
    out_percentage = ()  # percentage of tuples in the pipeline being in the output of this operator (will be summed up per operator type)
    right_card = ()  # cardinality of the right input

    # expression features for tablescans (and maybe selects)
    like_count = ()
    like_percentage = ()
    compare_count = ()
    compare_percentage = ()
    in_expression_count = ()
    in_expression_percentage = ()
    between_count = ()
    between_percentage = ()
    or_exp_count = ()
    or_exp_percentage = ()
    starts_with_count = ()
    starts_with_percentage = ()
    join_filter_count = ()
    false_count = ()

    @staticmethod
    def get_global_features() -> list["Feature"]:
        return [Feature.pipeline_scan_card, Feature.pipeline_sink_card]


class FeatureDim(AutoNumber):
    # const does not need a dimension
    scan = ()  # in_card, in_size
    sink = ()  # out_card, out_size, useful for scanning a sink operator
    input = ()  # in_percentage
    out = ()  # out_percentage
    right = ()  # right_percentage
    right_card = ()  # right cardinality
    input_card = ()  # input cardinality
    expressions = ()  # expressions
    empty_output = ()  # tablescans output zero tuples


class QualifiedFeature:
    pipeline_time_features: dict[OperatorType : dict[OperatorStage : list[FeatureDim]]] = {
        OperatorType.TableScan: {
            OperatorStage.Scan: [FeatureDim.scan, FeatureDim.out, FeatureDim.expressions, FeatureDim.empty_output]
        },
        OperatorType.InlineTable: {OperatorStage.Scan: [FeatureDim.scan, FeatureDim.out]},
        OperatorType.PipelineBreakerScan: {OperatorStage.Scan: [FeatureDim.scan, FeatureDim.out]},
        OperatorType.Temp: {OperatorStage.Build: [FeatureDim.sink, FeatureDim.input]},
        OperatorType.EarlyExecution: {OperatorStage.Scan: [FeatureDim.out]},
        OperatorType.Select: {OperatorStage.PassThrough: [FeatureDim.input, FeatureDim.out]},
        OperatorType.Map: {OperatorStage.PassThrough: [FeatureDim.input, FeatureDim.out]},
        OperatorType.MultiWayJoin: {
            OperatorStage.Build: [FeatureDim.sink, FeatureDim.input],
            OperatorStage.Scan: [FeatureDim.scan, FeatureDim.out],
        },
        OperatorType.HashJoin: {
            OperatorStage.Build: [FeatureDim.sink, FeatureDim.input],
            # during probing the size of the hash table might matter (but this is also not possible to add up :C)
            OperatorStage.Probe: [FeatureDim.input_card, FeatureDim.right, FeatureDim.out],
        },
        OperatorType.IndexNLJoin: {OperatorStage.Probe: [FeatureDim.input, FeatureDim.right_card, FeatureDim.out]},
        OperatorType.GroupJoin: {
            OperatorStage.Build: [FeatureDim.sink, FeatureDim.input],
            OperatorStage.Probe: [
                FeatureDim.sink,
                FeatureDim.right,
                FeatureDim.out,
            ],
            OperatorStage.Scan: [FeatureDim.scan, FeatureDim.out],
        },
        OperatorType.GroupBy: {
            OperatorStage.Build: [FeatureDim.sink, FeatureDim.input],
            OperatorStage.Scan: [FeatureDim.sink, FeatureDim.out],  # sink features seem to work better here
        },
        OperatorType.Sort: {
            OperatorStage.Build: [FeatureDim.sink, FeatureDim.input, FeatureDim.out],
            OperatorStage.Scan: [FeatureDim.scan, FeatureDim.out],
        },
        OperatorType.SetOperation: {
            OperatorStage.Build: [FeatureDim.sink, FeatureDim.input],
            OperatorStage.Scan: [FeatureDim.scan, FeatureDim.out],
            OperatorStage.PassThrough: [],
        },
        OperatorType.Window: {
            OperatorStage.Build: [FeatureDim.sink, FeatureDim.input],
            OperatorStage.Scan: [FeatureDim.scan, FeatureDim.out],
        },
        OperatorType.FileOutput: {OperatorStage.Build: [FeatureDim.sink, FeatureDim.input]},
        OperatorType.CsvWriter: {OperatorStage.Build: [FeatureDim.sink, FeatureDim.input]},
        OperatorType.AssertSingle: {OperatorStage.PassThrough: [FeatureDim.input]},
        OperatorType.EarlyProbe: {OperatorStage.PassThrough: [FeatureDim.out]},
    }

    def __init__(
        self, operator_type: Optional[OperatorType], operator_stage: Optional[OperatorStage], feature: Feature
    ):
        self.operator_type: Optional[OperatorType] = operator_type
        self.operator_stage: Optional[OperatorStage] = operator_stage
        self.feature: Feature = feature

    @staticmethod
    def get_dim_features(dim: FeatureDim) -> list[Feature]:
        if dim == FeatureDim.scan:
            return [Feature.in_card, Feature.in_size]
        if dim == FeatureDim.sink:
            return [Feature.out_card, Feature.out_size]
        if dim == FeatureDim.out:
            return [Feature.out_percentage]
        elif dim == FeatureDim.input:
            return [Feature.in_percentage]
        elif dim == FeatureDim.right:
            return [Feature.right_percentage]
        elif dim == FeatureDim.right_card:
            return [Feature.right_card]
        elif dim == FeatureDim.input_card:
            return [Feature.in_card]
        elif dim == FeatureDim.expressions:
            return [
                # Feature.like_count,
                Feature.like_percentage,
                # Feature.compare_count,
                Feature.compare_percentage,
                # Feature.in_expression_count,
                Feature.in_expression_percentage,
                # Feature.between_count,
                Feature.between_percentage,
                # Feature.or_exp_count,
                Feature.or_exp_percentage,
                # Feature.starts_with_count,
                Feature.starts_with_percentage,
                # Feature.join_filter_count,
                # Feature.false_count,
            ]
        elif dim == FeatureDim.empty_output:
            return [Feature.empty_output]
        assert False, "unhandled dimension"

    @staticmethod
    def enumerate_features() -> list["QualifiedFeature"]:
        """
        Create a list of all features that will be included by the feature vector
        """
        result = [
            # QualifiedFeature(None, None, Feature.pipeline_scan_card),
            # QualifiedFeature(None, None, Feature.pipeline_sink_card),
        ]
        for operator_type, stages in QualifiedFeature.pipeline_time_features.items():
            for stage, dims in stages.items():
                result.append(QualifiedFeature(operator_type, stage, Feature.const))
                for dim in dims:
                    for feature in QualifiedFeature.get_dim_features(dim):
                        result.append(QualifiedFeature(operator_type, stage, feature))
        return result

    @staticmethod
    def get_feature_index_lookup() -> dict["QualifiedFeature", int]:
        result = {}
        for i, f in enumerate(QualifiedFeature.enumerate_features()):
            result[f] = i
        return result

    @staticmethod
    def get_feature_lookup() -> dict[OperatorType, dict[OperatorStage, list["QualifiedFeature"]]]:
        """ """
        result = {}
        for feature in QualifiedFeature.enumerate_features():
            if feature.operator_type not in result:
                result[feature.operator_type] = {}
            if feature.operator_stage not in result[feature.operator_type]:
                result[feature.operator_type][feature.operator_stage] = []
            result[feature.operator_type][feature.operator_stage].append(feature)
        return result

    def get_name(self):
        return f"{self.operator_type.name}_{self.operator_stage.name}_{self.feature.name}"

    def __eq__(self, other: "QualifiedFeature"):
        return (
            other
            and self.operator_type == other.operator_type
            and self.operator_stage == other.operator_stage
            and self.feature == other.feature
        )

    def __ne__(self, other):
        return not self.__eq__(other)

    def __hash__(self):
        return hash((self.operator_type, self.operator_stage, self.feature))


class FeatureMapper:
    _lookup = QualifiedFeature.get_feature_lookup()
    _index_lookup = QualifiedFeature.get_feature_index_lookup()
    _features = QualifiedFeature.enumerate_features()
    n_features = len(_features)

    @staticmethod
    def get_features(op: OperatorType, stage: OperatorStage) -> list[QualifiedFeature]:
        if stage not in FeatureMapper._lookup[op]:
            return []
        return FeatureMapper._lookup[op][stage]

    def get_empty_feature_vector(self) -> np.ndarray:
        return np.zeros(self.n_features, dtype=float)

    def get_estimation_vector(self, phase: ExecutionPhase) -> np.ndarray:
        output_cardinality = phase.get_output_cardinality()
        input_cardinality = phase.get_input_cardinality()
        right_input_cardinality = phase.get_right_input_cardinality()
        output_size = phase.operator.output_tuple_size
        input_size = phase.operator.input_op.output_tuple_size if phase.operator.input_op is not None else 0
        input_percentage = phase.get_input_percentage()
        output_percentage = phase.get_output_percentage()
        right_percentage = phase.get_right_percentage()

        assert float(output_cardinality) or True
        assert float(input_cardinality) or True
        assert float(right_input_cardinality) or True
        if phase.operator.type == OperatorType.HashJoin:
            if phase.stage == OperatorStage.Build:
                output_cardinality = input_cardinality
                output_size = input_size
                output_percentage = input_percentage

        expressions = phase.operator.expressions

        features = {
            Feature.out_card: output_cardinality,
            Feature.in_card: input_cardinality,
            Feature.out_size: output_size,
            Feature.in_size: input_size,
            Feature.const: 1,
            Feature.in_percentage: input_percentage,
            Feature.out_percentage: output_percentage,
            Feature.right_percentage: right_percentage,
            Feature.right_card: right_input_cardinality,
            Feature.like_count: expressions.like_count,
            Feature.like_percentage: expressions.like_selectivity,
            Feature.compare_count: expressions.compare_count,
            Feature.compare_percentage: expressions.compare_selectivity,
            Feature.in_expression_count: expressions.in_expression_count,
            Feature.in_expression_percentage: expressions.in_expression_selectivity,
            Feature.between_count: expressions.between_count,
            Feature.between_percentage: expressions.between_selectivity,
            Feature.or_exp_count: expressions.or_expression_count,
            Feature.or_exp_percentage: expressions.or_selectivity,
            Feature.starts_with_count: expressions.starts_with_count,
            Feature.starts_with_percentage: expressions.starts_with_selectivity,
            Feature.join_filter_count: expressions.join_filter_count,
            Feature.false_count: expressions.false_count,
            Feature.empty_output: 1 if output_cardinality == 0 else 0,
        }
        assert len(Feature) == len(features) + len(Feature.get_global_features())

        result = self.get_empty_feature_vector()
        assert (
            len(self.get_features(phase.operator.type, phase.stage)) > 0
        ), f"no features for {phase.operator.type.name} - {phase.stage.name}"
        for f in self.get_features(phase.operator.type, phase.stage):
            value = features[f.feature]
            index = self._index_lookup[f]
            result[index] = value

        return result

    def get_estimation_matrix(self, query_plan: QueryPlan) -> np.ndarray:
        """
        get a feature vector for each operator in the query plan
        """
        row_vectors: list[np.ndarray] = []
        for pipeline in query_plan.pipelines:
            for op in pipeline.operators:
                row_vectors.append(self.get_estimation_vector(op))
        return np.vstack(row_vectors)

    def get_pipeline_estimation_matrix(self, query_plan: QueryPlan) -> np.ndarray:
        """
        get a feature vector for each pipeline in the query plan
        """
        result = []
        for pipeline in query_plan.pipelines:
            row_vectors: list[np.ndarray] = [self.get_empty_feature_vector()]
            for op in pipeline.operators:
                row_vectors.append(self.get_estimation_vector(op))
            pipeline_vector = np.sum(np.vstack(row_vectors), axis=0)
            result.append(pipeline_vector)
        return np.vstack(result)

    def get_pipeline_estimation_matrices(self, query_plan: QueryPlan) -> list[np.ndarray]:
        """
        get a feature vector for each operator in each pipeline
        build matrices by pipelines
        """
        result = []
        for pipeline in query_plan.pipelines:
            row_vectors: list[np.ndarray] = []
            for op in pipeline.operators:
                row_vectors.append(self.get_estimation_vector(op))
            pipeline_matrix = np.vstack(row_vectors)
            result.append(pipeline_matrix)
        return result

    def explain_features(self, query_plan: QueryPlan, pipeline: Optional[int] = None, verbose: bool = False):
        for i, pipeline_vector in enumerate(self.get_pipeline_estimation_matrix(query_plan)):
            if pipeline is None or i == pipeline:
                print(f" Pipeline{i} (scan: {query_plan.pipelines[i].get_pipeline_scan_cardinality()} tuples)")
                for n, v in zip(self.get_names(), pipeline_vector):
                    if verbose or v > 0.0:
                        print(f"  {n}: {v}")

    def get_single_estimation_vector(self, query_plan: QueryPlan):
        return np.sum(self.get_estimation_matrix(query_plan), axis=0)

    @staticmethod
    def get_names() -> list[str]:
        res = []
        for f in QualifiedFeature.enumerate_features():
            if f.operator_type is None:
                res.append(f"Global_{f.feature.name}")
            else:
                res.append(f"{f.operator_type.name}_{f.operator_stage.name}_{f.feature.name}")
        return res

    @staticmethod
    def get_pipeline_scan_sizes(query_plan: QueryPlan) -> np.ndarray:
        result = [p.get_pipeline_scan_cardinality() for p in query_plan.pipelines]
        return np.array(result)

    @staticmethod
    def get_portable_feature_encoding():
        result = {}
        for i, f in enumerate(QualifiedFeature.enumerate_features()):
            if f.operator_type.name not in result:
                result[f.operator_type.name] = {}
            if f.operator_stage.name not in result[f.operator_type.name]:
                result[f.operator_type.name][f.operator_stage.name] = {}
            result[f.operator_type.name][f.operator_stage.name][f.feature.name] = i
        result = json.dumps(result)
        result = result.lower()
        print(result)


def main():
    print("\n".join(f"{i} {n}" for i, n in enumerate(FeatureMapper.get_names())))
    FeatureMapper.get_portable_feature_encoding()


if __name__ == "__main__":
    main()
