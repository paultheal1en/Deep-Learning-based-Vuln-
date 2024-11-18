static void sdhci_do_data_transfer(void *opaque)

{

    SDHCIState *s = (SDHCIState *)opaque;



    SDHCI_GET_CLASS(s)->data_transfer(s);

}
