static void pc_dimm_realize(DeviceState *dev, Error **errp)

{

    PCDIMMDevice *dimm = PC_DIMM(dev);



    if (!dimm->hostmem) {

        error_setg(errp, "'" PC_DIMM_MEMDEV_PROP "' property is not set");

        return;

    }

    if ((nb_numa_nodes > 0) && (dimm->node >= nb_numa_nodes)) {

        error_setg(errp, "'DIMM property " PC_DIMM_NODE_PROP " has value %"

                   PRIu32 "' which exceeds the number of numa nodes: %d",

                   dimm->node, nb_numa_nodes);

        return;

    }

}
