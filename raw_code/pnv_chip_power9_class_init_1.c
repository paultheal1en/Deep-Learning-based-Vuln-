static void pnv_chip_power9_class_init(ObjectClass *klass, void *data)

{

    DeviceClass *dc = DEVICE_CLASS(klass);

    PnvChipClass *k = PNV_CHIP_CLASS(klass);



    k->cpu_model = "POWER9";

    k->chip_type = PNV_CHIP_POWER9;

    k->chip_cfam_id = 0x100d104980000000ull; /* P9 Nimbus DD1.0 */

    k->cores_mask = POWER9_CORE_MASK;

    k->core_pir = pnv_chip_core_pir_p9;


    dc->desc = "PowerNV Chip POWER9";

}