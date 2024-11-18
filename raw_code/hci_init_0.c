struct HCIInfo *hci_init(const char *str)

{

    char *endp;

    struct bt_scatternet_s *vlan = 0;



    if (!strcmp(str, "null"))

        /* null */

        return &null_hci;

    else if (!strncmp(str, "host", 4) && (str[4] == '\0' || str[4] == ':'))

        /* host[:hciN] */

        return bt_host_hci(str[4] ? str + 5 : "hci0");

    else if (!strncmp(str, "hci", 3)) {

        /* hci[,vlan=n] */

        if (str[3]) {

            if (!strncmp(str + 3, ",vlan=", 6)) {

                vlan = qemu_find_bt_vlan(strtol(str + 9, &endp, 0));

                if (*endp)

                    vlan = 0;

            }

        } else

            vlan = qemu_find_bt_vlan(0);

        if (vlan)

           return bt_new_hci(vlan);

    }



    fprintf(stderr, "qemu: Unknown bluetooth HCI `%s'.\n", str);



    return 0;

}
