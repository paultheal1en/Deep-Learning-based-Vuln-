bool tcg_target_deposit_valid(int ofs, int len)

{

    return (facilities & FACILITY_GEN_INST_EXT) != 0;

}
