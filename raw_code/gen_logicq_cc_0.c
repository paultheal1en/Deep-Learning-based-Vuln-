static void gen_logicq_cc(TCGv val)

{

    TCGv tmp = new_tmp();

    gen_helper_logicq_cc(tmp, val);

    gen_logic_CC(tmp);

    dead_tmp(tmp);

}
