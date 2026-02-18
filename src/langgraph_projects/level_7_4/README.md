Semantic
Facts
Bike model I have
Agent's Facts about a user

Episodic
Memories
Bike rides I took
Agent's Past agent actions

Procedural
Instructions
Motor skills
Agent’s system prompt
Agent's instructions for how to respond to user messages

# Store

## Namespace: (User, Profile)

```json
{
  "key": "profile",
  "value": {
    "name": "Matthew"
  }
}
```

## Namespace: (User, Memories)

```json
{
  "key": "memory_1",
  "value": {
    "dinner": "pizza"
  }
}
```

```json
{
  "key": "memory_2",
  "value": {
    "dinner": "pasta"
  }
}
```

# Store Architecture (Namespaced Memory)

The store is organized using **namespaces**.

A namespace acts like a folder path:

(User, Profile)
(User, Memories)

This allows you to group related data while keeping different categories separated.

## Conceptual Structure

```

Store
│
├── Namespace: (User, Profile)
│     └── key: "profile"
│         value: { "name": "Matthew" }
│
└── Namespace: (User, Memories)
├── key: "memory_1"
│     value: { "dinner": "pizza" }
│
└── key: "memory_2"
value: { "dinner": "pasta" }

```

---

## How Namespaces Work

• A **namespace** is a logical grouping of related data  
• It prevents collisions between similar keys  
• It allows structured storage for different memory types

Think of it like this:

```

(namespace_1)   profile → {...}
(namespace_2)   memory_1 → {...}
(namespace_2)   memory_2 → {...}

```

Even if two namespaces use the same key name, they remain separate because the namespace is part of the address.

---

## JSON Representation

### Namespace: (User, Profile)

```json
{
  "namespace": ["User", "Profile"],
  "key": "profile",
  "value": {
    "name": "Matthew"
  }
}
```

### Namespace: (User, Memories)

```json
{
  "namespace": ["User", "Memories"],
  "key": "memory_1",
  "value": {
    "dinner": "pizza"
  }
}
```

```json
{
  "namespace": ["User", "Memories"],
  "key": "memory_2",
  "value": {
    "dinner": "pasta"
  }
}
```

---

## Why This Matters

This structure enables:

• Clean separation of user profile data
• Multiple episodic memories per user
• Scalable organization for agents
• Future expansion (e.g., Preferences, Settings, History)

---

💡 In short:

Namespace = category (directory)
Key = identifier within that category
Value = stored data

```

If you'd like, I can also provide a **Mermaid diagram version** for visual rendering in GitHub.
```
