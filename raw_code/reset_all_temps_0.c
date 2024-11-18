static void reset_all_temps(int nb_temps)

{

    int i;

    for (i = 0; i < nb_temps; i++) {

        temps[i].state = TCG_TEMP_UNDEF;

        temps[i].mask = -1;

    }

}
