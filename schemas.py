"""
Database Schemas

Define your MongoDB collection schemas here using Pydantic models.
Each Pydantic model represents a collection in your database.
Model name is converted to lowercase for the collection name:
- User -> "user" collection
- Product -> "product" collection
- BlogPost -> "blogpost" collection
"""

from pydantic import BaseModel, Field
from typing import Optional, Dict, Any, List

# Core app schemas

class Document(BaseModel):
    """Documents collection schema"""
    title: str = Field(..., description="Original file name or title")
    pages: Optional[int] = Field(None, ge=0, description="Number of pages")
    table_count: Optional[int] = Field(0, ge=0, description="Number of extracted tables")
    metadata: Optional[Dict[str, Any]] = Field(default_factory=dict, description="Arbitrary metadata")

class Chunk(BaseModel):
    """Text chunks with lightweight embeddings for semantic search"""
    doc_id: str = Field(..., description="Reference to the parent document _id")
    text: str = Field(..., description="Chunk text")
    page: Optional[int] = Field(None, ge=1, description="Page number if known")
    embedding: Dict[str, float] = Field(default_factory=dict, description="Sparse term-weight map")
    metadata: Optional[Dict[str, Any]] = Field(default_factory=dict, description="Extra info about the chunk")

# Example schemas retained (not used by app but kept for viewer compatibility)
class User(BaseModel):
    name: str = Field(..., description="Full name")
    email: str = Field(..., description="Email address")
    address: str = Field(..., description="Address")
    age: Optional[int] = Field(None, ge=0, le=120, description="Age in years")
    is_active: bool = Field(True, description="Whether user is active")

class Product(BaseModel):
    title: str = Field(..., description="Product title")
    description: Optional[str] = Field(None, description="Product description")
    price: float = Field(..., ge=0, description="Price in dollars")
    category: str = Field(..., description="Product category")
    in_stock: bool = Field(True, description="Whether product is in stock")
