static void tcg_target_qemu_prologue (TCGContext *s)

{

    int i, frame_size;

#ifndef __APPLE__

    uint64_t addr;

#endif



    frame_size = 0

        + 8                     /* back chain */

        + 8                     /* CR */

        + 8                     /* LR */

        + 8                     /* compiler doubleword */

        + 8                     /* link editor doubleword */

        + 8                     /* TOC save area */

        + TCG_STATIC_CALL_ARGS_SIZE

        + ARRAY_SIZE (tcg_target_callee_save_regs) * 8

        ;

    frame_size = (frame_size + 15) & ~15;



#ifndef __APPLE__

    /* First emit adhoc function descriptor */

    addr = (uint64_t) s->code_ptr + 24;

    tcg_out32 (s, addr >> 32); tcg_out32 (s, addr); /* entry point */

    s->code_ptr += 16;          /* skip TOC and environment pointer */

#endif



    /* Prologue */

    tcg_out32 (s, MFSPR | RT (0) | LR);

    tcg_out32 (s, STDU | RS (1) | RA (1) | (-frame_size & 0xffff));

    for (i = 0; i < ARRAY_SIZE (tcg_target_callee_save_regs); ++i)

        tcg_out32 (s, (STD

                       | RS (tcg_target_callee_save_regs[i])

                       | RA (1)

                       | (i * 8 + 48 + TCG_STATIC_CALL_ARGS_SIZE)

                       )

            );

    tcg_out32 (s, STD | RS (0) | RA (1) | (frame_size + 16));



#ifdef CONFIG_USE_GUEST_BASE

    if (GUEST_BASE) {

        tcg_out_movi (s, TCG_TYPE_I64, TCG_GUEST_BASE_REG, GUEST_BASE);

        tcg_regset_set_reg(s->reserved_regs, TCG_GUEST_BASE_REG);

    }

#endif



    tcg_out32 (s, MTSPR | RS (3) | CTR);

    tcg_out32 (s, BCCTR | BO_ALWAYS);



    /* Epilogue */

    tb_ret_addr = s->code_ptr;



    for (i = 0; i < ARRAY_SIZE (tcg_target_callee_save_regs); ++i)

        tcg_out32 (s, (LD

                       | RT (tcg_target_callee_save_regs[i])

                       | RA (1)

                       | (i * 8 + 48 + TCG_STATIC_CALL_ARGS_SIZE)

                       )

            );

    tcg_out32 (s, LD | RT (0) | RA (1) | (frame_size + 16));

    tcg_out32 (s, MTSPR | RS (0) | LR);

    tcg_out32 (s, ADDI | RT (1) | RA (1) | frame_size);

    tcg_out32 (s, BCLR | BO_ALWAYS);

}
