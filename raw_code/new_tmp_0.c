static TCGv new_tmp(void)

{

    TCGv tmp;

    if (num_temps == MAX_TEMPS)

        abort();



    if (GET_TCGV(temps[num_temps]))

      return temps[num_temps++];



    tmp = tcg_temp_new(TCG_TYPE_I32);

    temps[num_temps++] = tmp;

    return tmp;

}
