static TCGv_i32 new_tmp(void)

{

    num_temps++;

    return tcg_temp_new_i32();

}
