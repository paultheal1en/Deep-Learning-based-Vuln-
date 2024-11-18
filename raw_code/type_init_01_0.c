type_init(pflash_cfi01_register_types)



pflash_t *pflash_cfi01_register(hwaddr base,

                                DeviceState *qdev, const char *name,

                                hwaddr size,

                                BlockDriverState *bs,

                                uint32_t sector_len, int nb_blocs,

                                int bank_width, uint16_t id0, uint16_t id1,

                                uint16_t id2, uint16_t id3, int be)

{

    DeviceState *dev = qdev_create(NULL, TYPE_CFI_PFLASH01);



    if (bs && qdev_prop_set_drive(dev, "drive", bs)) {

        abort();

    }

    qdev_prop_set_uint32(dev, "num-blocks", nb_blocs);

    qdev_prop_set_uint64(dev, "sector-length", sector_len);

    qdev_prop_set_uint8(dev, "width", bank_width);

    qdev_prop_set_uint8(dev, "big-endian", !!be);

    qdev_prop_set_uint16(dev, "id0", id0);

    qdev_prop_set_uint16(dev, "id1", id1);

    qdev_prop_set_uint16(dev, "id2", id2);

    qdev_prop_set_uint16(dev, "id3", id3);

    qdev_prop_set_string(dev, "name", name);

    qdev_init_nofail(dev);



    sysbus_mmio_map(SYS_BUS_DEVICE(dev), 0, base);

    return CFI_PFLASH01(dev);

}
