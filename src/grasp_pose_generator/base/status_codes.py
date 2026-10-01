"""统一状态码定义。"""


class StatusCode:
    """抓取位姿生成阶段统一状态码。"""

    OK = "ok"
    INVALID_INPUT = "invalid_input"
    CONFIG_ERROR = "config_error"
    ADAPTER_NOT_FOUND = "adapter_not_found"
    ADAPTER_UNAVAILABLE = "adapter_unavailable"
    NATIVE_RUNTIME_ERROR = "native_runtime_error"
    MISSING_WORLD_TRANSFORM = "missing_world_transform"
    NO_VALID_DEPTH = "no_valid_depth"
    NO_VALID_ROI_POINTS = "no_valid_roi_points"
    NO_CANDIDATE = "no_candidate"
    VISIBLE_BUT_UNGRASPABLE = "visible_but_ungraspable"
    NOT_IMPLEMENTED = "not_implemented"


SUCCESS_CODES = {StatusCode.OK}
