# Update _detect_header to use full rows
@@
-def _detect_header(first_row: List[str]) -> bool:
-    """Detect if the first row is a header.
-
-    The heuristic looks at each column individually. If the first row value
-    is non‑numeric and all subsequent rows in that column are numeric, we
-    treat the first row as a header. If this condition is true for *any*
-    column, we consider the file to have a header.
-
-    This approach works for typical CSVs where headers are strings and data
-    rows contain numbers, while still handling cases where the first row is
-    data that happens to contain strings.
-    """
-    if not first_row:
-        return False
-    for v in first_row:
-        if v == "":
-            continue
-        if not (_is_int(v) or _is_float(v)):
-            return True
-    return False
+def _detect_header(rows: List[List[str]]) -> bool:
+    """Detect if the first row is a header.
+
+    The heuristic examines each column. If the first row value is non‑numeric
+    and **all** subsequent rows in that column are numeric, the column is
+    considered a header column. If *any* column satisfies this condition, the
+    file is treated as having a header.
+    """
+    if not rows:
+        return False
+    first_row = rows[0]
+    # Transpose to iterate columns
+    columns = list(zip(*rows))
+    for idx, col in enumerate(columns):
+        first_val = col[0]
+        if first_val == "":
+            continue
+        if not (_is_int(first_val) or _is_float(first_val)):
+            # Check if all subsequent values are numeric
+            if all((_is_int(v) or _is_float(v)) for v in col[1:]):
+                return True
+    return False
*** End Patch