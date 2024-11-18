void tcg_target_qemu_prologue (TCGContext *s)

{

    int i, frame_size;



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



    tcg_out32 (s, MFSPR | RT (0) | LR);

    tcg_out32 (s, STDU | RS (1) | RA (1) | (-frame_size & 0xffff));

    for (i = 0; i < ARRAY_SIZE (tcg_target_callee_save_regs); ++i)

        tcg_out32 (s, (STD

                       | RS (tcg_target_callee_save_regs[i])

                       | RA (1)

                       | (i * 8 + 48 + TCG_STATIC_CALL_ARGS_SIZE)

                       )

            );

    tcg_out32 (s, STD | RS (0) | RA (1) | (frame_size + 20));

    tcg_out32 (s, STD | RS (2) | RA (1) | (frame_size + 40));



    tcg_out32 (s, MTSPR | RS (3) | CTR);

    tcg_out32 (s, BCCTR | BO_ALWAYS);

    tb_ret_addr = s->code_ptr;



    for (i = 0; i < ARRAY_SIZE (tcg_target_callee_save_regs); ++i)

        tcg_out32 (s, (LD

                       | RT (tcg_target_callee_save_regs[i])

                       | RA (1)

                       | (i * 8 + 48 + TCG_STATIC_CALL_ARGS_SIZE)

                       )

            );

    tcg_out32 (s, LD | RT (0) | RA (1) | (frame_size + 20));

    tcg_out32 (s, LD | RT (2) | RA (1) | (frame_size + 40));

    tcg_out32 (s, MTSPR | RS (0) | LR);

    tcg_out32 (s, ADDI | RT (1) | RA (1) | frame_size);

    tcg_out32 (s, BCLR | BO_ALWAYS);

}
