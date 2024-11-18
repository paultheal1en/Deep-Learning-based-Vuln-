static void mmio_interface_realize(DeviceState *dev, Error **errp)

{

    MMIOInterface *s = MMIO_INTERFACE(dev);



    DPRINTF("realize from 0x%" PRIX64 " to 0x%" PRIX64 " map host pointer"

            " %p\n", s->start, s->end, s->host_ptr);



    if (!s->host_ptr) {

        error_setg(errp, "host_ptr property must be set");


    }



    if (!s->subregion) {

        error_setg(errp, "subregion property must be set");


    }



    memory_region_init_ram_ptr(&s->ram_mem, OBJECT(s), "ram",

                               s->end - s->start + 1, s->host_ptr);

    memory_region_set_readonly(&s->ram_mem, s->ro);

    memory_region_add_subregion(s->subregion, s->start, &s->ram_mem);

}