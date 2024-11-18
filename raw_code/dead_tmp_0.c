static void dead_tmp(TCGv tmp)

{

    int i;

    num_temps--;

    i = num_temps;

    if (GET_TCGV(temps[i]) == GET_TCGV(tmp))

        return;



    /* Shuffle this temp to the last slot.  */

    while (GET_TCGV(temps[i]) != GET_TCGV(tmp))

        i--;

    while (i < num_temps) {

        temps[i] = temps[i + 1];

        i++;

    }

    temps[i] = tmp;

}
