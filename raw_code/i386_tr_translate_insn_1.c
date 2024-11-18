static void i386_tr_translate_insn(DisasContextBase *dcbase, CPUState *cpu)

{

    DisasContext *dc = container_of(dcbase, DisasContext, base);

    target_ulong pc_next = disas_insn(dc, cpu);



    if (dc->tf || (dc->base.tb->flags & HF_INHIBIT_IRQ_MASK)) {

        /* if single step mode, we generate only one instruction and

           generate an exception */

        /* if irq were inhibited with HF_INHIBIT_IRQ_MASK, we clear

           the flag and abort the translation to give the irqs a

           chance to happen */

        dc->base.is_jmp = DISAS_TOO_MANY;

    } else if ((dc->base.tb->cflags & CF_USE_ICOUNT)

               && ((dc->base.pc_next & TARGET_PAGE_MASK)

                   != ((dc->base.pc_next + TARGET_MAX_INSN_SIZE - 1)

                       & TARGET_PAGE_MASK)

                   || (dc->base.pc_next & ~TARGET_PAGE_MASK) == 0)) {

        /* Do not cross the boundary of the pages in icount mode,

           it can cause an exception. Do it only when boundary is

           crossed by the first instruction in the block.

           If current instruction already crossed the bound - it's ok,

           because an exception hasn't stopped this code.

         */

        dc->base.is_jmp = DISAS_TOO_MANY;

    } else if ((pc_next - dc->base.pc_first) >= (TARGET_PAGE_SIZE - 32)) {

        dc->base.is_jmp = DISAS_TOO_MANY;

    }



    dc->base.pc_next = pc_next;

}
