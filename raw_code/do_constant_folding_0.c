static TCGArg do_constant_folding(int op, TCGArg x, TCGArg y)

{

    TCGArg res = do_constant_folding_2(op, x, y);

#if TCG_TARGET_REG_BITS == 64

    if (op_bits(op) == 32) {

        res &= 0xffffffff;

    }

#endif

    return res;

}
