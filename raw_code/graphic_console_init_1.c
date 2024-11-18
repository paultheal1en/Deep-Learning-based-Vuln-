QemuConsole *graphic_console_init(DeviceState *dev, uint32_t head,

                                  const GraphicHwOps *hw_ops,

                                  void *opaque)

{

    Error *local_err = NULL;

    int width = 640;

    int height = 480;

    QemuConsole *s;

    DisplayState *ds;



    ds = get_alloc_displaystate();

    trace_console_gfx_new();

    s = new_console(ds, GRAPHIC_CONSOLE);

    s->hw_ops = hw_ops;

    s->hw = opaque;

    if (dev) {

        object_property_set_link(OBJECT(s), OBJECT(dev),

                                 "device", &local_err);

        object_property_set_int(OBJECT(s), head,

                                "head", &local_err);

    }



    s->surface = qemu_create_displaysurface(width, height);

    return s;

}
