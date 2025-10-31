# OT algorithm placeholder
# For simplicity, we implement a very basic character insertion/deletion OT

class OTHandler:
    @staticmethod
    def transform(op, pending_ops, client_version, server_version):
        """Transform the incoming op against pending ops.
        For this demo, we ignore complex transformations and just return op.
        """
        return op

    @staticmethod
    def apply(content, op):
        """Apply an operation to the content.
        op format: {"type": "insert"|"delete", "pos": int, "text": str}
        """
        if op["type"] == "insert":
            pos = op["pos"]
            text = op["text"]
            return content[:pos] + text + content[pos:]
        elif op["type"] == "delete":
            pos = op["pos"]
            length = op.get("length", 1)
            return content[:pos] + content[pos+length:]
        else:
            return content
