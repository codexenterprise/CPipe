import time
from cpipe.module.logic import Logic
from cpipe.module.node import Node
from cagent.core.tools import ToolResult


class my_node(Logic):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.mystate = 1
        
    @Node.agent_tool(name="add",
                       description="Add two numbers.",
                       parameters={"type": "function", "properties": {"a": {"type": "int", "description": "The first number."}, "b": {"type": "int", "description": "The second number."}}, "required": ["a", "b"]})
    def add(self, a, b) -> ToolResult:
        if a is None or b is None:
            return ToolResult(success=False, output=None, error="a or b is None")
        if not isinstance(a, int) or not isinstance(b, int):
            return ToolResult(success=False, output=None, error="a or b is not int")
        return ToolResult(success=True, output=a + b)
    
    @Node.event("event_get_node_state")
    def get_node_state_event(self, data):
        """
        get Node state
        """
        print(f"get_node_state_event: {data}")
        return self.mystate
    
    @Node.agent_tool(name="get_node_state",
                       description="Get the state of the node.",
                       parameters={})
    def get_node_state(self) -> ToolResult:
        return ToolResult(success=True, output=f"The current status of the node is: {self.event_send('get_node_state', 5555)}")

    def _start(self):

        while True:
            new_cdata = self.get_frames()
            # set mcp data
            self.mystate += 1

            time.sleep(1)

            self.queue.put_cdata(new_cdata)
