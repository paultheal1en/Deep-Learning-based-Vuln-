static int bt_hci_parse(const char *str)

{

    struct HCIInfo *hci;

    bdaddr_t bdaddr;



    if (nb_hcis >= MAX_NICS) {

        fprintf(stderr, "qemu: Too many bluetooth HCIs (max %i).\n", MAX_NICS);

        return -1;

    }



    hci = hci_init(str);

    if (!hci)

        return -1;



    bdaddr.b[0] = 0x52;

    bdaddr.b[1] = 0x54;

    bdaddr.b[2] = 0x00;

    bdaddr.b[3] = 0x12;

    bdaddr.b[4] = 0x34;

    bdaddr.b[5] = 0x56 + nb_hcis;

    hci->bdaddr_set(hci, bdaddr.b);



    hci_table[nb_hcis++] = hci;



    return 0;

}
