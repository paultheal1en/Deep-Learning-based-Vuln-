static void mmubooke_create_initial_mapping(CPUPPCState *env,

                                     target_ulong va,

                                     hwaddr pa)

{

    ppcemb_tlb_t *tlb = &env->tlb.tlbe[0];



    tlb->attr = 0;

    tlb->prot = PAGE_VALID | ((PAGE_READ | PAGE_WRITE | PAGE_EXEC) << 4);

    tlb->size = 1 << 31; /* up to 0x80000000  */

    tlb->EPN = va & TARGET_PAGE_MASK;

    tlb->RPN = pa & TARGET_PAGE_MASK;

    tlb->PID = 0;



    tlb = &env->tlb.tlbe[1];

    tlb->attr = 0;

    tlb->prot = PAGE_VALID | ((PAGE_READ | PAGE_WRITE | PAGE_EXEC) << 4);

    tlb->size = 1 << 31; /* up to 0xffffffff  */

    tlb->EPN = 0x80000000 & TARGET_PAGE_MASK;

    tlb->RPN = 0x80000000 & TARGET_PAGE_MASK;

    tlb->PID = 0;

}
