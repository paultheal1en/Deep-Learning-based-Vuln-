static void pnv_chip_power8nvl_class_init(ObjectClass *klass, void *data)

{

    DeviceClass *dc = DEVICE_CLASS(klass);

    PnvChipClass *k = PNV_CHIP_CLASS(klass);



    k->cpu_model = "POWER8NVL";

    k->chip_type = PNV_CHIP_POWER8NVL;

    k->chip_cfam_id = 0x120d304980000000ull;  /* P8 Naples DD1.0 */

    k->cores_mask = POWER8_CORE_MASK;

    k->core_pir = pnv_chip_core_pir_p8;


    dc->desc = "PowerNV Chip POWER8NVL";

}