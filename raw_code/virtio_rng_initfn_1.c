static void virtio_rng_initfn(Object *obj)

{

    VirtIORNG *vrng = VIRTIO_RNG(obj);



    object_property_add_link(obj, "rng", TYPE_RNG_BACKEND,

                             (Object **)&vrng->conf.rng, NULL);

}
