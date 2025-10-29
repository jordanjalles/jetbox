"""
Inventory management system.

Provides:
- Product dataclass
- Inventory class with add, remove, search, low-stock alerts, CSV export
"""

from __future__ import annotations

import csv
from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, Iterable, List, Optional


@dataclass(eq=True, frozen=True)
class Product:
    """Represents a product in the inventory.

    Attributes
    ----------
    sku : str
        Unique identifier for the product.
    name : str
        Human‑readable name.
    price : float
        Unit price.
    quantity : int
        Current stock level.
    """

    sku: str
    name: str
    price: float
    quantity: int = field(default=0)

    def __post_init__(self):
        if self.quantity < 0:
            raise ValueError("Quantity cannot be negative")
        if self.price < 0:
            raise ValueError("Price cannot be negative")


class Inventory:
    """Container for :class:`Product` objects.

    The inventory is stored in a dictionary keyed by SKU for O(1) access.
    """

    def __init__(self, low_stock_threshold: int = 5) -> None:
        self._products: Dict[str, Product] = {}
        self.low_stock_threshold = low_stock_threshold

    # ------------------------------------------------------------------
    # CRUD operations
    # ------------------------------------------------------------------
    def add_product(self, product: Product) -> None:
        """Add a product to the inventory.

        If a product with the same SKU already exists, its quantity is
        increased by the new product's quantity.
        """
        if product.sku in self._products:
            existing = self._products[product.sku]
            new_qty = existing.quantity + product.quantity
            self._products[product.sku] = Product(
                sku=existing.sku,
                name=existing.name,
                price=existing.price,
                quantity=new_qty,
            )
        else:
            self._products[product.sku] = product

    def remove_product(self, sku: str, quantity: int = 1) -> None:
        """Remove a quantity of a product.

        Raises ``KeyError`` if the SKU is unknown.
        Raises ``ValueError`` if the quantity to remove is greater than
        available stock.
        """
        if sku not in self._products:
            raise KeyError(f"SKU {sku} not found")
        product = self._products[sku]
        if quantity > product.quantity:
            raise ValueError(
                f"Cannot remove {quantity} units; only {product.quantity} in stock"
            )
        new_qty = product.quantity - quantity
        if new_qty == 0:
            del self._products[sku]
        else:
            self._products[sku] = Product(
                sku=product.sku,
                name=product.name,
                price=product.price,
                quantity=new_qty,
            )

    def search_by_name(self, query: str) -> List[Product]:
        """Return a list of products whose name contains *query* (case‑insensitive)."""
        q = query.lower()
        return [p for p in self._products.values() if q in p.name.lower()]

    # ------------------------------------------------------------------
    # Utility methods
    # ------------------------------------------------------------------
    def low_stock_alerts(self) -> List[Product]:
        """Return a list of products below the low‑stock threshold."""
        return [p for p in self._products.values() if p.quantity < self.low_stock_threshold]

    def export_to_csv(self, file_path: str | Path) -> None:
        """Export the inventory to a CSV file.

        The CSV contains the columns: sku, name, price, quantity.
        """
        path = Path(file_path)
        path.parent.mkdir(parents=True, exist_ok=True)
        with path.open("w", newline="", encoding="utf-8") as f:
            writer = csv.writer(f)
            writer.writerow(["sku", "name", "price", "quantity"])
            for p in self._products.values():
                writer.writerow([p.sku, p.name, f"{p.price:.2f}", p.quantity])

    # ------------------------------------------------------------------
    # Convenience
    # ------------------------------------------------------------------
    def __iter__(self):
        return iter(self._products.values())

    def __len__(self) -> int:
        return len(self._products)

    def __repr__(self) -> str:
        return f"<Inventory {len(self)} products>"


# ----------------------------------------------------------------------
# Example usage (uncomment to run as a script)
# ----------------------------------------------------------------------
if __name__ == "__main__":
    inv = Inventory(low_stock_threshold=3)
    inv.add_product(Product("SKU001", "Widget", 9.99, 10))
    inv.add_product(Product("SKU002", "Gadget", 14.99, 2))
    inv.add_product(Product("SKU003", "Thingamajig", 4.50, 5))

    print("All products:")
    for p in inv:
        print(p)

    print("\nLow stock alerts:")
    for p in inv.low_stock_alerts():
        print(p)

    inv.export_to_csv("inventory_export.csv")
    print("\nExported to inventory_export.csv")
"""
