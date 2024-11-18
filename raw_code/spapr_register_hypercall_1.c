void spapr_register_hypercall(target_ulong opcode, spapr_hcall_fn fn)

{

    spapr_hcall_fn old_fn;



    assert(opcode <= MAX_HCALL_OPCODE);

    assert((opcode & 0x3) == 0);



    old_fn = hypercall_table[opcode / 4];



    assert(!old_fn || (fn == old_fn));



    hypercall_table[opcode / 4] = fn;

}
