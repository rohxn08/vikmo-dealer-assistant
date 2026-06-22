import os
import uuid
import datetime
import pandas as pd
from pydantic import BaseModel
from typing import List

# Resolve catalogue path relative to this script
BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
CATALOGUE_PATH = os.path.join(BASE_DIR, "catalogue.csv")

if not os.path.exists(CATALOGUE_PATH):
    raise FileNotFoundError(f"catalogue.csv not found at {CATALOGUE_PATH}")

df = pd.read_csv(CATALOGUE_PATH).set_index("sku")

# --- Tool 1: check_stock ---
def check_stock(sku: str) -> dict:
    """Check availability, price, and details of a specific product SKU.
    
    Args:
        sku: The unique product identifier (e.g. 'BRK-1042').
        
    Returns:
        A dictionary containing the SKU, product name, current stock level, and price in INR.
    """
    sku = str(sku).strip().upper()
    if sku not in df.index:
        return {"error": f"SKU {sku} not found"}
    row = df.loc[sku]
    return {
        "sku": sku,
        "name": str(row["name"]),
        "stock": int(row["stock"]),
        "price_inr": int(row["price_inr"])
    }

# --- Tool 2: find_parts_by_vehicle ---
def find_parts_by_vehicle(vehicle: str) -> list[dict]:
    """Find parts from the catalogue that fit a specific vehicle make or model.
    
    Args:
        vehicle: The make or model name of the vehicle (e.g. 'Yamaha FZ', 'KTM Duke 390').
        
    Returns:
        A list of matching products, each with SKU, name, price in INR, and stock level.
    """
    vehicle_query = str(vehicle).strip().lower()
    # Check contains vehicle, case-insensitive
    matches = df[df["vehicle_fitment"].str.contains(vehicle_query, case=False, na=False)]
    
    # Return sku as well
    results = []
    for sku, row in matches.head(10).iterrows():
        results.append({
            "sku": str(sku),
            "name": str(row["name"]),
            "price_inr": int(row["price_inr"]),
            "stock": int(row["stock"])
        })
    return results

# --- Tool 3: create_order (Pydantic validated) ---
class OrderItem(BaseModel):
    sku: str
    quantity: int

class OrderRequest(BaseModel):
    dealer_name: str
    items: List[OrderItem]

class OrderConfirmation(BaseModel):
    order_id: str
    dealer_name: str
    items: List[dict]
    total_inr: int
    status: str
    created_at: str

def create_order(dealer_name: str, items: list[dict]) -> dict:
    """Place an order for one or more product items on behalf of a dealer.
    
    Args:
        dealer_name: Name of the dealership placing the order.
        items: List of items to order. Each item is a dictionary with keys 'sku' and 'quantity'.
        
    Returns:
        Order confirmation details, including an order ID, line item subtotals, and total price.
    """
    try:
        request = OrderRequest(dealer_name=dealer_name,
                               items=[OrderItem(**i) for i in items])
    except Exception as e:
        return {"error": f"Invalid order request structure: {e}"}

    line_items, total = [], 0
    for item in request.items:
        sku = item.sku.strip().upper()
        if sku not in df.index:
            return {"error": f"SKU {sku} not found"}
        
        row = df.loc[sku]
        stock_available = int(row["stock"])
        if item.quantity <= 0:
            return {"error": f"Quantity for {sku} must be greater than zero"}
            
        subtotal = int(row["price_inr"]) * item.quantity
        total += subtotal
        line_items.append({
            "sku": sku,
            "name": str(row["name"]),
            "qty": item.quantity,
            "subtotal_inr": subtotal
        })

    confirmation = OrderConfirmation(
        order_id=f"ORD-{uuid.uuid4().hex[:8].upper()}",
        dealer_name=request.dealer_name,
        items=line_items,
        total_inr=total,
        status="confirmed",
        created_at=datetime.datetime.utcnow().isoformat()
    )
    return confirmation.model_dump()
