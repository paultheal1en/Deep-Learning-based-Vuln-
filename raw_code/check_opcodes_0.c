static int check_opcodes(MMCO *mmco1, MMCO *mmco2, int n_mmcos)

{

    int i;



    for (i = 0; i < n_mmcos; i++) {

        if (mmco1[i].opcode != mmco2[i].opcode)

            return -1;

    }



    return 0;

}
