sdhci_buff_access_is_sequential(SDHCIState *s, unsigned byte_num)

{

    if ((s->data_count & 0x3) != byte_num) {

        ERRPRINT("Non-sequential access to Buffer Data Port register"

                "is prohibited\n");

        return false;

    }

    return true;

}
