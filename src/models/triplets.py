"models/triplets.py"
"""
Data models for Knowledge Graph entities
"""
from dataclasses import dataclass
from typing import Optional


@dataclass
class Triplet:
    """
    Represents a knowledge graph triplet (subject, predicate, object)
    
    Example:
        >>> t = Triplet("Alice", "works_at", "Acme Corp")
        >>> print(t)
        (Alice) --[works_at]--> (Acme Corp)
    """
    subject: str
    predicate: str
    object: str
    
    def __str__(self) -> str:
        return f"({self.subject}) --[{self.predicate}]--> ({self.object})"
    
    def __repr__(self) -> str:
        return f"Triplet(subject='{self.subject}', predicate='{self.predicate}', object='{self.object}')"
    
    def to_dict(self) -> dict:
        """Convert to dictionary"""
        return {
            "subject": self.subject,
            "predicate": self.predicate,
            "object": self.object
        }
    
    @classmethod
    def from_dict(cls, data: dict) -> "Triplet":
        """Create from dictionary"""
        return cls(
            subject=data["subject"],
            predicate=data["predicate"],
            object=data["object"]
        )


@dataclass
class Entity:
    """Represents an entity in the knowledge graph"""
    id: str
    embedding: Optional[list[float]] = None
    metadata: Optional[dict] = None
    
    def __str__(self) -> str:
        return f"Entity({self.id})"