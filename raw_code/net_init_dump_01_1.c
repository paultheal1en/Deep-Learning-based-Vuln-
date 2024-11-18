int net_init_dump(QemuOpts *opts, const char *name, VLANState *vlan)

{

    int len;

    const char *file;

    char def_file[128];



    assert(vlan);



    file = qemu_opt_get(opts, "file");

    if (!file) {

        snprintf(def_file, sizeof(def_file), "qemu-vlan%d.pcap", vlan->id);

        file = def_file;

    }



    len = qemu_opt_get_size(opts, "len", 65536);



    return net_dump_init(vlan, "dump", name, file, len);

}
