static KVMSlot *kvm_lookup_slot(KVMState *s, target_phys_addr_t start_addr)

{

    int i;



    for (i = 0; i < ARRAY_SIZE(s->slots); i++) {

        KVMSlot *mem = &s->slots[i];



        if (start_addr >= mem->start_addr &&

            start_addr < (mem->start_addr + mem->memory_size))

            return mem;

    }



    return NULL;

}
