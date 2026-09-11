def audit_documents(docs: list[Document]) -> None:
    """Print a quality report on loaded documents."""
    total = len(docs)
    empty = sum(1 for d in docs if not d.page_content.strip())
    very_short = sum(1 for d in docs if 0 < len(d.page_content) < 50)
    
    print(f"Total documents: {total}")
    print(f"Empty documents: {empty} ({100*empty/total:.1f}%)")
    print(f"Very short (<50 chars): {very_short} ({100*very_short/total:.1f}%)")
    
    # Check metadata completeness
    missing_source = sum(1 for d in docs if 'source' not in d.metadata)
    print(f"Missing 'source' metadata: {missing_source}")
    
    # Show a random sample
    import random
    sample = random.choice(docs)
    print(f"\nSample document:")
    print(f"  Metadata: {sample.metadata}")
    print(f"  Content preview: {sample.page_content[:200]}")

audit_documents(documents)