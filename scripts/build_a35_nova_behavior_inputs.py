"""Build one reproducible, disabled curriculum proposal, without model APIs."""
import hashlib
import json
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from evaluation.a35_nova_behavior_data import reference_answer

SLOTS = {
    "arithmetic": ["a35-arithmetic", "a35-nova-transfer-arithmetic"],
    "probability": ["a35-probability", "a35-nova-transfer-probability"],
    "geometry": ["a35-rectangle", "a35-nova-transfer-geometry"],
    "equation": ["a35-equation", "a35-nova-transfer-integer_root"],
    "predicate": ["a35-even-filter", "a35-modulo-filter"],
    "copy": ["a35-copy", "a35-nova-copy-isolation"],
    "increment": ["a35-increment", "a35-nova-transfer-increment"],
    "string_root": ["a35-next-stage", "a35-nova-transfer-string_root"],
    "median": ["a35-median", "a35-nova-transfer-array_root"],
}


def build():
    rows, reviews = [], []

    def add(group, split, variant, question, operation, inputs, structure):
        rid = f"a35-nova-behavior-{group}-{variant}-{split}"
        reference = {"operation": operation, "inputs": inputs}
        answer = reference_answer(reference)
        rows.append({"id": rid, "split": split, "messages": [
            {"role": "user", "content": question},
            {"role": "assistant", "content": json.dumps(answer, ensure_ascii=False, separators=(",", ":"))}]})
        reviews.append({"id": rid, "split": split, "group": group,
            "replaces": SLOTS[group][variant]+"-"+split,
            "structure": structure, "reference": reference, "code_fixture": None,
            "criteria": [], "provenance": "project_authored_synthetic", "policy_review": "conformant"})

    def py(group, split, variant, description, source, structure):
        add(group, split, variant, description+"\nPython 3:\n"+source+"\nSerialize result as exactly one JSON value.",
            "python_trace", {"source": source}, structure)

    add("arithmetic", "train", 0,
        "Audit stock: opening 86; seven lots each contain 8 small plus 5 large items, four lots each contain 2 small plus 4 large; losses are 18 and 9. Add within each lot before multiplying. Return JSON per_lot_subtotals in lot order, incoming, removed, balance.",
        "ledger", {"opening":86,"groups":[[7,8,5],[4,2,4]],"removals":[18,9],"checked":True}, "grouped_subtotals_two_product_ledger_audit")
    add("arithmetic", "train", 1,
        "Workshop planning starts with 41 pieces. Eight packets each supply 15 bolts and 2 nuts; three packets each supply 6 bolts and 3 nuts. After combining all pieces, 26 are scrapped. Give only an object containing usable.",
        "ledger", {"opening":41,"groups":[[8,15,2],[3,6,3]],"removals":[26],"checked":False,"key":"usable"}, "two_kind_packet_inventory_adjustment")
    add("arithmetic", "validation", 0,
        "An imaginary account carries 37 credits. Twelve credits packages pay 7 base credits and 4 bonus credits apiece; five packages pay 9 base and 5 bonus. Debit thirty-nine and sixteen. Supply balance, incoming, removed and per_lot_subtotals in package order as one JSON object.",
        "ledger", {"opening":37,"groups":[[12,7,4],[5,9,5]],"removals":[39,16],"checked":True}, "base_bonus_package_reconciliation")
    add("arithmetic", "validation", 1,
        "A venue counts 58 existing seats. Six new rows have 16 regular and 7 folding seats each; two rows have 8 regular and 11 folding each. Remove 47 damaged seats and 13 reserved seats from the combined inventory. Respond with JSON available only.",
        "ledger", {"opening":58,"groups":[[6,16,7],[2,8,11]],"removals":[47,13],"checked":False,"key":"available"}, "mixed_seating_grouped_product_removals")

    add("probability", "train", 0,
        "A shuffled deck contains four amber, five violet and two silver cards. Draw amber then violet then silver, returning nothing. Audit the event: eligible and population arrays before each selection, and fraction as reduced [numerator,denominator]. JSON object only.",
        "ordered_probability", {"counts":{"amber":4,"violet":5,"silver":2},"sequence":["amber","violet","silver"],"checked":True}, "three_color_ordered_count_audit")
    add("probability", "train", 1,
        "There are 6 copper, 3 white and 2 black discs in an urn. Three successive unreplaced selections must be white, copper, white, in precisely that order. Provide only {\"fraction\":[reduced numerator,reduced denominator]}.",
        "ordered_probability", {"counts":{"copper":6,"white":3,"black":2},"sequence":["white","copper","white"],"checked":False}, "nonadjacent_repeated_color")
    add("probability", "validation", 0,
        "Randomly inspect labels from a sealed collection: five north, two south, four east. All inspected labels stay outside the collection. The ordered observation sought is east, east, south. In one JSON object report population totals and eligible counts at each inspection plus fraction [n,d], reduced.",
        "ordered_probability", {"counts":{"north":5,"south":2,"east":4},"sequence":["east","east","south"],"checked":True}, "adjacent_repetition_three_labels")
    add("probability", "validation", 1,
        "An opaque box holds 2 maple, 4 pine, 3 oak markers. Record the chance of the successive sample oak, maple, pine, oak when sampled markers are set aside. Express the exact chance using a JSON object whose fraction property is a pair of coprime integers.",
        "ordered_probability", {"counts":{"maple":2,"pine":4,"oak":3},"sequence":["oak","maple","pine","oak"],"checked":False}, "four_step_three_species_event")

    add("geometry", "train", 0,
        "A garden boundary uses 52 metres of fence plus a 4-metre gate. Its long side is 21 metres. A 4-by-3 pond removes usable ground. Return JSON sides [length,width], perimeter_check, usable area. Count the gate as boundary.",
        "geometry", {"fence":52,"gate":4,"length":21,"cutout":[4,3],"checked":True}, "gate_boundary_dimension_area_check")
    add("geometry", "train", 1,
        "For a rectangular enclosure, fencing spans 66 units and the doorway another 6. The long edge measures 25. Deduct the area of a 3-by-5 equipment pad. Give only usable in a JSON object.",
        "geometry", {"fence":66,"gate":6,"length":25,"cutout":[3,5],"checked":False}, "enclosure_access_gap_with_pad")
    add("geometry", "validation", 0,
        "A plot's side-length ratio is 5:3. Its wire border measures 72 units; two openings total 8. A rectangular hole is 2 by 7. Verify geometry using an object: perimeter_check, sides in ratio order, usable. The openings remain part of the closed boundary.",
        "geometry", {"fence":72,"gate":8,"ratio":[5,3],"cutout":[2,7],"checked":True}, "ratio_plot_two_openings")
    add("geometry", "validation", 1,
        "An exhibit floor is rectangular with edges in ratio 4:1. Rope covers 55 metres of its outline and entrance gaps cover 5 more. A 5-by-2 kiosk occupies the floor. Emit a JSON object identifying usable square metres.",
        "geometry", {"fence":55,"gate":5,"ratio":[4,1],"cutout":[5,2],"checked":False}, "scaled_exhibit_ratio_kiosk")

    add("equation", "train", 0,
        "Find integer u in 4(u-3)=2u+18. Produce one JSON object with u and the computed left and right sides after substitution, under left and right.",
        "equation", {"a":4,"b":-12,"c":2,"d":18,"key":"u","checked":True}, "distributed_unknown_both_sides_substitution")
    add("equation", "train", 1,
        "Two ticket formulas agree: 7(t+2)-5 and 3t+45. Determine the integer ticket variable. Reply solely with the JSON property t and its value.",
        "equation", {"a":7,"b":9,"c":3,"d":45,"key":"t","checked":False}, "ticket_formula_balance")
    add("equation", "validation", 0,
        "Check the balance 5-3v=2(v+5). Name v as an integer and evaluate both expressions independently. Your JSON object must have left, right and v; equal substituted expressions certify the root.",
        "equation", {"a":-3,"b":5,"c":2,"d":10,"key":"v","checked":True}, "negative_coefficient_balance_check")
    add("equation", "validation", 1,
        "For a fictional calibration, 2(3q-4)+7=4q+17. Give the calibrated integer using {\"q\":integer}, with no extra properties.",
        "equation", {"a":6,"b":-1,"c":4,"d":17,"key":"q","checked":False}, "nested_calibration_distribution")

    py("predicate", "train", 0, "Audit selection separately from transformation; record retained source indices as well as outputs.",
        "records=[{'n':-4,'ok':True},{'n':3,'ok':True},{'n':0,'ok':True},{'n':-4,'ok':True},{'n':8,'ok':False}]\nkept=[]\noutputs=[]\nfor index,row in enumerate(records):\n    if row['ok'] and row['n']%2==0:\n        kept.append(index)\n        outputs.append(row['n']+3)\nresult={'indices':kept,'outputs':outputs}", "record_flags_even_loop_original_predicate")
    py("predicate", "train", 1, "Keep source order and duplicates. The eligibility condition applies before the output calculation.",
        "readings=[-6,4,0,9,-6,5]\nresult=[]\nfor value in readings:\n    if value%3!=0:\n        continue\n    result.append(value*value-2)", "continue_nonmultiple_square_transform")
    py("predicate", "validation", 0, "A paired quality flag can reject an otherwise eligible sample. Return the selected position/value pairs.",
        "samples=[5,0,-8,5,6]\nflags=[True,False,True,True,True]\nresult=[[i,n-1] for i,(n,flag) in enumerate(zip(samples,flags)) if flag and n%4==1]", "zip_flags_nonzero_remainder_pairs")
    py("predicate", "validation", 1, "Select valid signed multiples, transform survivors, and preserve multiplicity. Empty selections are valid arrays.",
        "packets=[{'units':-9},{'units':4},{'units':-9},{'units':0},{'units':12}]\nresult=[row['units']//3+1 for row in packets if row['units']%3==0 and row['units']<1]", "record_comprehension_compound_filter")

    py("copy", "train", 0, "Track outer containers separately from their shared inner lists; report both final views.",
        "original=[['r'],['s']]\nduplicate=original.copy()\nduplicate.append(['u'])\nduplicate[0].append('t')\nresult={'original':original,'duplicate':duplicate}", "nested_outer_copy_shared_element_audit")
    py("copy", "train", 1, "A removal from the copied outer list and an inner mutation have different effects. Report original only.",
        "original=[[4,5],[8]]\nmirror=original.copy()\nmirror.pop()\nmirror[0].append(6)\nresult=original", "outer_pop_inner_append_original_view")
    py("copy", "validation", 0, "Evaluate mapping-copy ownership and shared payloads. Return the resulting original and replica mappings.",
        "original={'items':[7],'label':'old'}\nreplica=dict(original)\nreplica['label']='new'\nreplica['items'].extend([2,9])\nresult={'original':original,'replica':replica}", "dict_constructor_copy_shared_payload")
    py("copy", "validation", 1, "Distinguish sharing an object from rebinding a variable; return the object still reached through source.",
        "source={'values':[3,8]}\nalias=source\nalias['values'].pop()\nalias={'values':[20]}\nresult=source", "mapping_alias_mutation_then_rebinding")

    py("increment", "train", 0, "Audit every accumulator event. Return the before/after pairs and final counters.",
        "stock={'left':6,'right':4}\nchanges=[('left',-2),('right',3),('left',5)]\ntrace=[]\nfor key,delta in changes:\n    before=stock[key]\n    stock[key]+=delta\n    trace.append([before,stock[key]])\nresult={'trace':trace,'stock':stock}", "signed_repeated_event_accumulator_trace")
    py("increment", "train", 1, "Read both operands from the current dictionary before the compound assignment; report its final state.",
        "totals={'red':8,'blue':5}\nfor key,other in [('red','blue'),('blue','red')]:\n    totals[key]+=totals[other]\nresult=totals", "cross_key_ordered_compound_updates")
    py("increment", "validation", 0, "A counter may not exist yet. Preserve updates in their stated order, with absent counters beginning at zero.",
        "counts={}\nlog=[{'name':'oak','delta':5},{'name':'pine','delta':-2},{'name':'oak','delta':-3}]\nfor item in log:\n    name=item['name']\n    counts[name]=counts.get(name,0)+item['delta']\nresult=counts", "get_default_record_event_replay")
    py("increment", "validation", 1, "The alias and original reach the same mapping. Later right-hand operands see earlier updates.",
        "state={'north':7,'south':2}\nalias=state\nalias['north']-=state['south']\nstate['south']*=alias['north']\nresult=state", "aliased_subtract_then_multiply")

    py("string_root", "train", 0, "Only an unfinished job whose required resources are available is eligible. Return the single selected name as a JSON string, never a wrapper.",
        "requires={'seal':['frame','key'],'pack':['frame','box'],'send':['seal','pack']}\navailable={'frame','box'}\nready=[name for name,deps in requires.items() if set(deps)<=available]\nresult=ready[0]", "resource_gated_fork_string")
    py("string_root", "train", 1, "A state table is authoritative. Produce the resulting state as a double-quoted JSON string alone.",
        "transitions={'hot':{'pause':'cooldown','finish':'done'},'queued':{'start':'hot'}}\ncurrent='hot'\nevent='pause'\nresult=transitions[current][event]", "nested_state_event_scalar_lookup")
    py("string_root", "validation", 0, "Find a task that remains unfinished and has no missing prerequisite. Exactly one is available. Serialize its identifier itself.",
        "completed={'setup','permit'}\ntasks=[('scan',{'setup','permit'}),('ship',{'scan','seal'}),('seal',{'key'})]\nresult=next(name for name,deps in tasks if name not in completed and not (deps-completed))", "set_difference_generator_ready_identifier")
    py("string_root", "validation", 1, "The ordered rules below choose the first matching label. Return that label as JSON text, without an object or array.",
        "temperature=12\nrules=[(temperature<0,'freeze'),(temperature<15,'idle'),(True,'active')]\nselected=[]\nfor matches,label in rules:\n    if matches and not selected:\n        selected.append(label)\nresult=selected[0]", "priority_rule_single_label_selection")

    py("median", "train", 0, "Expand this frequency table and expose a checkable sorted sample. Return sorted_values, count and median in one JSON object.",
        "table=[(2,3),(11,2),(17,1)]\nvalues=[]\nfor value,count in table:\n    values.extend([value]*count)\nvalues=sorted(values)\nn=len(values)\nresult={'sorted_values':values,'count':n,'median':(values[n//2-1]+values[n//2])/2}", "frequency_expansion_even_median_audit")
    py("median", "train", 1, "Exclude missing and explicitly invalid readings before taking the middle. Return median as an object property.",
        "readings=[{'v':8,'ok':True},{'v':None,'ok':False},{'v':2,'ok':True},{'v':99,'ok':False},{'v':6,'ok':True}]\nvalues=[]\nfor reading in readings:\n    if reading['ok'] and reading['v'] is not None:\n        values.append(reading['v'])\nvalues=sorted(values)\nresult={'median':values[len(values)//2]}", "valid_record_odd_median_object")
    py("median", "validation", 0, "A sample is already sorted in descending order. Supply its count, ascending sorted_values, and the exact midpoint statistic; do not average all observations.",
        "descending=[23,18,9,7,4,1]\nascending=descending[::-1]\nmid=len(ascending)//2\nresult={'count':len(ascending),'sorted_values':ascending,'median':sum(ascending[mid-1:mid+1])/2}", "descending_even_pair_slice_statistic")
    py("median", "validation", 1, "Derive the location statistic from retained positive measurements. Keep the required object root.",
        "observations=[-1,14,5,0,8,30,9]\npositive=sorted(x for x in observations if x>0)\nresult={'median':positive[(len(positive)-1)//2]}", "positive_generator_odd_midpoint")

    data_path = "data/a35-nova-behavior.v1.draft.jsonl"
    review_path = "data/a35-nova-behavior-review.v1.json"
    (ROOT/data_path).write_text("".join(json.dumps(r,ensure_ascii=False,separators=(",",":"))+"\n" for r in rows))
    review = {"schema_version":1,"status":"owner_delegated_policy_conformance_review",
        "provenance":{"source":"project_authored_synthetic","license":"CC0-1.0",
                      "third_party_training_text":False,"customer_data":False,"secrets":False},
        "model_calls_made":0,"quality_improvement_proved":False,"records":reviews}
    (ROOT/review_path).write_text(json.dumps(review,indent=2)+"\n")
    pins = [data_path,review_path,"prompts/kova-identity.v5.txt","prompts/kova-nova-task-checks.v2.txt",
            "evaluations/model-quality-suite.v1.json","config/a35-quality-policy.v1.json",
            "config/a35-nova-screen.v1.json"]
    manifest = {"schema_version":1,"candidate":"a35-nova-behavior-v10", "family":"kova-nova",
        "scope":"offline_review_only","quality_status":"UNMEASURED",
        "parent_source":"f63e24c36f30cf26b584dc903a5df95e16c36e72",
        "dataset_path":data_path,"review_path":review_path,
        "file_sha256":{p:hashlib.sha256((ROOT/p).read_bytes()).hexdigest() for p in pins},
        "model_calls_made":0,"new_training_runs":0,"new_allocations":0,
        "execution_authorized":False,"training_authorized":False,"paid_inference_authorized":False,
        "deployment_authorized":False,"merge_authorized":False,"quality_improvement_proved":False,
        "consumed_grant_reuse_authorized":False,
        "integration":"Separate offline prepared pack only; existing paid launcher and authorization packages unchanged.",
        "preserved_recipe":{"epochs":3,"optimizer_steps":30,"training_cap_seconds":600,
                            "train_records":73,"validation_records":61},
        "quality_gates":{"strict":36,"manual_cases":14,"manual_criteria":48,
                         "required_repetitions":3,"strict_aggregate":108,"averaging_failures_allowed":False,
                         "applicable_identity_safety_grounding_failures_allowed":0},
        "tokenizer_and_real_trl_masks":"NOT_RUN_FOR_NEW_ROWS"}
    (ROOT/"config/a35-nova-behavior-correction.v1.json").write_text(json.dumps(manifest,indent=2)+"\n")
    print(json.dumps({"rows":len(rows),"review_records":len(reviews),"model_calls":0,"execution_authorized":False}))


if __name__ == "__main__":
    import argparse
    argparse.ArgumentParser(description=__doc__).parse_args()
    build()
