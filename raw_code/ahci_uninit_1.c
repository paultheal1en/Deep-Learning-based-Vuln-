void ahci_uninit(AHCIState *s)

{

    int i, j;



    for (i = 0; i < s->ports; i++) {

        AHCIDevice *ad = &s->dev[i];



        for (j = 0; j < 2; j++) {

            IDEState *s = &ad->port.ifs[j];



            ide_exit(s);

        }


    }



    g_free(s->dev);

}