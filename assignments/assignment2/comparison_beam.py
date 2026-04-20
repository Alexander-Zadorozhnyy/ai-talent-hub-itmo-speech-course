import json

def load_json(file_path, idx: int = 0):
    """Load JSON file and return labels and results"""
    with open(file_path, 'r', encoding='utf-8') as f:
        data = json.load(f)
    return data['labels'], data['results'][idx]  # Take first result from each

def extract_predictions(result, method_name):
    """Extract predictions from result"""
    if method_name in result['eval']:
        return result['eval'][method_name]['predictions']
    return {}

def get_word_context(text, changed_positions, window=3):
    """Extract context around changed words"""
    words = text.split()
    if not changed_positions:
        return text
    
    # Get all positions that are within window of any changed word
    context_positions = set()
    for pos in changed_positions:
        for offset in range(-window, window + 1):
            context_positions.add(pos + offset)
    
    # Filter valid positions
    context_positions = [p for p in context_positions if 0 <= p < len(words)]
    context_positions.sort()
    
    # Build context string with markers
    result_words = []
    for i, word in enumerate(words):
        if i in context_positions:
            if i in changed_positions:
                result_words.append(f"**{word}**")
            else:
                result_words.append(word)
        elif i < context_positions[0] - 1:
            # Skip words before first context
            if not result_words or result_words[-1] != "...":
                result_words.append("...")
        elif i > context_positions[-1] + 1:
            # Skip words after last context
            if result_words and result_words[-1] != "...":
                result_words.append("...")
    
    return ' '.join(result_words)

def find_changed_words(original, modified):
    """Find positions of changed words between two strings"""
    orig_words = original.split()
    mod_words = modified.split()
    
    changed = []
    min_len = min(len(orig_words), len(mod_words))
    
    for i in range(min_len):
        if orig_words[i] != mod_words[i]:
            changed.append(i)
    
    # If lengths differ, consider all extra words as changed
    if len(mod_words) > min_len:
        changed.extend(range(min_len, len(mod_words)))
    if len(orig_words) > min_len:
        changed.extend(range(min_len, len(orig_words)))
    
    return changed

def format_with_context(reference, beam, fusion, rescore, fusion_changed, rescore_changed):
    """Format text with context around changed words"""
    
    # For beam (baseline), show context where either method changed
    beam_changed_positions = set()
    if fusion_changed:
        beam_changed_positions.update(find_changed_words(beam, fusion))
    if rescore_changed:
        beam_changed_positions.update(find_changed_words(beam, rescore))
    
    beam_context = get_word_context(beam, beam_changed_positions)
    
    # For fusion, show context around its changes
    fusion_changed_positions = find_changed_words(beam, fusion) if fusion_changed else []
    fusion_context = get_word_context(fusion, fusion_changed_positions) if fusion_changed else get_word_context(fusion, [])
    
    # For rescore, show context around its changes
    rescore_changed_positions = find_changed_words(beam, rescore) if rescore_changed else []
    rescore_context = get_word_context(rescore, rescore_changed_positions) if rescore_changed else get_word_context(rescore, [])
    
    return beam_context, fusion_context, rescore_context

def find_differences(beam_preds, fusion_preds, rescore_preds, labels):
    """Find samples where LM methods differ from beam search"""
    differences = []
    
    for idx in range(len(labels)):
        beam_text = beam_preds.get(str(idx), '').strip()
        fusion_text = fusion_preds.get(str(idx), '').strip()
        rescore_text = rescore_preds.get(str(idx), '').strip()
        reference = labels.get(str(idx), '').strip()
        
        # Check if either LM method changed the hypothesis
        if fusion_text != beam_text or rescore_text != beam_text:
            # Get context for all texts
            fusion_changed = fusion_text != beam_text
            rescore_changed = rescore_text != beam_text
            
            beam_context, fusion_context, rescore_context = format_with_context(
                reference, beam_text, fusion_text, rescore_text, 
                fusion_changed, rescore_changed
            )
            
            differences.append({
                'idx': idx,
                'reference': reference,
                'beam': beam_context,
                'fusion': fusion_context,
                'rescore': rescore_context,
                'full_beam': beam_text,
                'full_fusion': fusion_text,
                'full_rescore': rescore_text,
                'fusion_changed': fusion_changed,
                'rescore_changed': rescore_changed,
                'fusion_equals_rescore': fusion_text == rescore_text
            })
    
    return differences

def main():
    # Load all three JSON files
    print("Loading JSON files...")
    labels, beam_result = load_json('./results/beam_width_comparison.json')
    _, fusion_result = load_json('./results/beam_lm_alpha_beta_comparison.json', 17)
    _, rescore_result = load_json('./results/beam_lm_rescore_comparison.json', 5)
    
    # Extract predictions
    beam_preds = extract_predictions(beam_result, 'beam')
    fusion_preds = extract_predictions(fusion_result, 'beam_lm')
    rescore_preds = extract_predictions(rescore_result, 'beam_lm_rescore')
    
    print(f"Loaded {len(beam_preds)} samples\n")
    
    # Find differences
    differences = find_differences(beam_preds, fusion_preds, rescore_preds, labels)
    
    # Separate cases where SF and rescore disagree
    disagreements = [d for d in differences if d['fusion_changed'] and d['rescore_changed'] and not d['fusion_equals_rescore']]
    both_agree = [d for d in differences if d['fusion_changed'] and d['rescore_changed'] and d['fusion_equals_rescore']]
    fusion_only = [d for d in differences if d['fusion_changed'] and not d['rescore_changed']]
    rescore_only = [d for d in differences if d['rescore_changed'] and not d['fusion_changed']]
    
    # Print comparison table with context
    print("="*120)
    print("QUALITATIVE COMPARISON: LM Methods vs. Plain Beam Search")
    print("="*120)
    print("\n**Bold** indicates changed words. Showing context (±3 words) around changes.")
    print("'...' indicates skipped words.\n")
    print("| # | Reference (context) | Beam (context) | Shallow Fusion (context) | Rescoring (context) |")
    print("|---|---------------------|----------------|--------------------------|---------------------|")
    
    for i, diff in enumerate(differences[:10]):
        # Get reference context around changes (use beam changes as reference)
        beam_changed_positions = set()
        if diff['fusion_changed']:
            beam_changed_positions.update(find_changed_words(diff['full_beam'], diff['full_fusion']))
        if diff['rescore_changed']:
            beam_changed_positions.update(find_changed_words(diff['full_beam'], diff['full_rescore']))
        
        ref_context = get_word_context(diff['reference'], beam_changed_positions)
        
        print(f"| {i+1} | {ref_context[:60]}... | {diff['beam'][:60]}... | {diff['fusion'][:60]}... | {diff['rescore'][:60]}... |")
    
    # Print examples where SF and rescore disagree with full context
    print("\n" + "="*120)
    print("CASES WHERE SHALLOW FUSION AND RESCORING DISAGREE")
    print("="*120)
    
    if disagreements:
        print(f"\nFound {len(disagreements)} cases where the two LM methods produced different results\n")
        
        for i, diff in enumerate(disagreements[:5]):  # Show first 5
            print(f"\n--- Example {i+1} (Sample ID: {diff['idx']}) ---")
            print(f"Reference:  {diff['reference']}")
            print(f"\nBeam:       {diff['full_beam']}")
            print(f"            {get_word_context(diff['full_beam'], find_changed_words(diff['full_beam'], diff['full_fusion']) + find_changed_words(diff['full_beam'], diff['full_rescore']))}")
            
            if diff['fusion_changed']:
                fusion_changed_pos = find_changed_words(diff['full_beam'], diff['full_fusion'])
                print(f"\nFusion:     {diff['full_fusion']}")
                print(f"            {get_word_context(diff['full_fusion'], fusion_changed_pos)}")
            
            if diff['rescore_changed']:
                rescore_changed_pos = find_changed_words(diff['full_beam'], diff['full_rescore'])
                print(f"\nRescore:    {diff['full_rescore']}")
                print(f"            {get_word_context(diff['full_rescore'], rescore_changed_pos)}")
            
            # Show where the methods differ
            fusion_pos = set(find_changed_words(diff['full_beam'], diff['full_fusion']))
            rescore_pos = set(find_changed_words(diff['full_beam'], diff['full_rescore']))
            
            if fusion_pos != rescore_pos:
                print(f"\nDifference: Shallow Fusion changed positions {sorted(fusion_pos)}")
                print(f"            Rescoring changed positions {sorted(rescore_pos)}")
            
            print("-" * 80)
    
    # Print examples where only one method changed
    print("\n" + "="*120)
    print("EXAMPLES WHERE ONLY ONE LM METHOD CHANGED THE HYPOTHESIS")
    print("="*120)
    
    print("\n--- Shallow Fusion Only Examples (first 3) ---")
    for i, diff in enumerate(fusion_only[:3]):
        print(f"\n{i+1}. Sample {diff['idx']}:")
        print(f"   Reference: {diff['reference']}")
        print(f"   Beam:      {diff['full_beam']}")
        fusion_changed_pos = find_changed_words(diff['full_beam'], diff['full_fusion'])
        print(f"   Fusion:    {diff['full_fusion']}")
        print(f"              {get_word_context(diff['full_fusion'], fusion_changed_pos)}")
        print(f"   Rescore:   {diff['full_rescore']} [UNCHANGED]")
    
    print("\n--- Rescoring Only Examples (first 3) ---")
    for i, diff in enumerate(rescore_only[:3]):
        print(f"\n{i+1}. Sample {diff['idx']}:")
        print(f"   Reference: {diff['reference']}")
        print(f"   Beam:      {diff['full_beam']}")
        print(f"   Fusion:    {diff['full_fusion']} [UNCHANGED]")
        rescore_changed_pos = find_changed_words(diff['full_beam'], diff['full_rescore'])
        print(f"   Rescore:   {diff['full_rescore']}")
        print(f"              {get_word_context(diff['full_rescore'], rescore_changed_pos)}")
    
    # Print statistics
    print("\n" + "="*120)
    print("STATISTICS")
    print("="*120)
    print(f"\nTotal samples analyzed: {len(beam_preds)}")
    print(f"Samples where at least one LM method changed: {len(differences)} ({len(differences)/len(beam_preds)*100:.1f}%)")
    print(f"\n- Shallow Fusion changed: {len(fusion_only) + len(both_agree) + len(disagreements)}")
    print(f"- Rescoring changed: {len(rescore_only) + len(both_agree) + len(disagreements)}")
    print(f"- Both changed (agreeing): {len(both_agree)}")
    print(f"- Both changed (disagreeing): {len(disagreements)}")
    print(f"- Only Shallow Fusion changed: {len(fusion_only)}")
    print(f"- Only Rescoring changed: {len(rescore_only)}")

if __name__ == "__main__":
    main()