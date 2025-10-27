from _typeshed import Incomplete

REGISTER: int
UNREGISTER: int

class _ResourceList:
    """Acl resources of current application
    This class provide register inferace of acl resource, when application
    exit, all register resource will release befor acl.rt.reset_device to
    avoid program abnormal 
    """
    _instance_lock: Incomplete
    resources: Incomplete
    def __init__(self) -> None: ...
    def __new__(cls, *args, **kwargs): ...
    def register(self, resource) -> None:
        """Resource register interface
        Args:
            resource: object with acl resource, the object must be has
                      method destroy()
        """
    def unregister(self, resource) -> None:
        """Resource unregister interface
        If registered resource release by self and no need _ResourceList 
        release, the resource object should unregister self
        Args:
            resource: registered resource
        """
    def destroy(self) -> None:
        """Destroy all register resource"""

resource_list: Incomplete

class CACLModel:
    """
    wrap acl model inference interface, include input dataset construction,
    execute, and output transform to numpy array
    Attributes:
        model_path: om offline mode file path
    """
    logger: Incomplete
    device_id: Incomplete
    context: Incomplete
    stream: Incomplete
    run_mode: Incomplete
    _copy_policy: Incomplete
    _model_path: Incomplete
    _load_type: Incomplete
    _model_id: Incomplete
    _input_num: int
    _input_buffer: Incomplete
    _input_dataset: Incomplete
    _output_dataset: Incomplete
    _model_desc: Incomplete
    _output_size: int
    _is_batch_size_dynamic: bool
    _is_destroyed: bool
    def __init__(self, model_path, device_id: int = 0, load_type: int = 0) -> None: ...
    def init(self):
        """
        init resource
        """
    def copy_data_device_to_host(self, device_data, data_size):
        """Copy device data to host
        Args:
            device_data: data that to be copyed
            data_size: data size
        Returns:
            None: copy failed
            others: host data which copy from device_data
        """
    def copy_data_device_to_device(self, device_data, data_size):
        """Copy device data to device
        Args:
            device_data: data that to be copyed
            data_size: data size
        Returns:
            None: copy failed
            others: device data which copy from device_data
        """
    def copy_data_host_to_device(self, host_data, data_size):
        """Copy host data to device
        Args:
            host_data: data that to be copyed
            data_size: data size
        Returns:
            None: copy failed
            others: device data which copy from host_data
        """
    def copy_data_host_to_host(self, host_data, data_size):
        """Copy host data to host
        Args:
            host_data: data that to be copyed
            data_size: data size
        Returns:
            None: copy failed
            others: host data which copy from host_data
        """
    def _init_resource(self): ...
    def _gen_output_dataset(self, ouput_num): ...
    def _init_input_buffer(self) -> None: ...
    def _gen_input_dataset(self, input_list): ...
    def _parse_input_data(self, input_data, index): ...
    def _copy_input_to_device(self, input_ptr, size, index): ...
    def is_support_dynamic_batch_size(self): ...
    def _set_dynamic_batch_size(self, batch): ...
    def execute(self, input_list):
        """
        inference input data
        Args:
            input_list: input data list, support AclLiteImage,
            numpy array and {'data': ,'size':} dict
        returns:
            inference result data, which is a numpy array list,
            each corresponse to a model output
        """
    def _output_dataset_to_numpy(self, batch_size): ...
    def _gen_output_tensor(self): ...
    def _release_dataset(self, dataset, free_memory: bool = False) -> None: ...
    def _release_databuffer(self, data_buffer, free_memory: bool = False) -> None: ...
    def destroy(self) -> None:
        """
        release resource of model inference
        Args:
            null
        Returns:
            null
        """
    def __del__(self) -> None: ...
