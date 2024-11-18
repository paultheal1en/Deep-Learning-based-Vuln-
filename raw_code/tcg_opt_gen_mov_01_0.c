static void tcg_opt_gen_mov(TCGContext *s, TCGOp *op, TCGArg *args,

                            TCGArg dst, TCGArg src)

{

    if (temps_are_copies(dst, src)) {

        tcg_op_remove(s, op);

        return;

    }



    if (temp_is_const(src)) {

        tcg_opt_gen_movi(s, op, args, dst, temps[src].val);

        return;

    }



    TCGOpcode new_op = op_to_mov(op->opc);

    tcg_target_ulong mask;



    op->opc = new_op;



    reset_temp(dst);

    mask = temps[src].mask;

    if (TCG_TARGET_REG_BITS > 32 && new_op == INDEX_op_mov_i32) {

        /* High bits of the destination are now garbage.  */

        mask |= ~0xffffffffull;

    }

    temps[dst].mask = mask;



    assert(!temp_is_const(src));



    if (s->temps[src].type == s->temps[dst].type) {

        temps[dst].next_copy = temps[src].next_copy;

        temps[dst].prev_copy = src;

        temps[temps[dst].next_copy].prev_copy = dst;

        temps[src].next_copy = dst;

        temps[dst].is_const = false;

    }



    args[0] = dst;

    args[1] = src;

}
