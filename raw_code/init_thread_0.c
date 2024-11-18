static inline void init_thread(struct target_pt_regs *regs, struct image_info *infop)

{

    target_long *stack = (void *)infop->start_stack;

    memset(regs, 0, sizeof(*regs));

    regs->ARM_cpsr = 0x10;

    regs->ARM_pc = infop->entry;

    regs->ARM_sp = infop->start_stack;

    regs->ARM_r2 = tswapl(stack[2]); /* envp */

    regs->ARM_r1 = tswapl(stack[1]); /* argv */

    /* XXX: it seems that r0 is zeroed after ! */

    //    regs->ARM_r0 = tswapl(stack[0]); /* argc */

}
