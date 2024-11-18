static void tcg_out_setcond2(TCGContext *s, const TCGArg *args,

                             const int *const_args)

{

    TCGArg new_args[6];

    int label_true, label_over;



    memcpy(new_args, args+1, 5*sizeof(TCGArg));



    if (args[0] == args[1] || args[0] == args[2]

        || (!const_args[3] && args[0] == args[3])

        || (!const_args[4] && args[0] == args[4])) {

        /* When the destination overlaps with one of the argument

           registers, don't do anything tricky.  */

        label_true = label_arg(gen_new_label());

        label_over = label_arg(gen_new_label());



        new_args[5] = label_true;

        tcg_out_brcond2(s, new_args, const_args+1, 1);



        tcg_out_movi(s, TCG_TYPE_I32, args[0], 0);

        tcg_out_jxx(s, JCC_JMP, label_over, 1);

        tcg_out_label(s, label_true, s->code_ptr);



        tcg_out_movi(s, TCG_TYPE_I32, args[0], 1);

        tcg_out_label(s, label_over, s->code_ptr);

    } else {

        /* When the destination does not overlap one of the arguments,

           clear the destination first, jump if cond false, and emit an

           increment in the true case.  This results in smaller code.  */



        tcg_out_movi(s, TCG_TYPE_I32, args[0], 0);



        label_over = label_arg(gen_new_label());

        new_args[4] = tcg_invert_cond(new_args[4]);

        new_args[5] = label_over;

        tcg_out_brcond2(s, new_args, const_args+1, 1);



        tgen_arithi(s, ARITH_ADD, args[0], 1, 0);

        tcg_out_label(s, label_over, s->code_ptr);

    }

}
