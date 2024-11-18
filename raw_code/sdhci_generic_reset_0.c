static void sdhci_generic_reset(DeviceState *ds)

{

    SDHCIState *s = SDHCI(ds);

    SDHCI_GET_CLASS(s)->reset(s);

}
