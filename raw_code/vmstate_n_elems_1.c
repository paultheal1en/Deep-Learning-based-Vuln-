static int vmstate_n_elems(void *opaque, VMStateField *field)

{

    int n_elems = 1;



    if (field->flags & VMS_ARRAY) {

        n_elems = field->num;

    } else if (field->flags & VMS_VARRAY_INT32) {

        n_elems = *(int32_t *)(opaque+field->num_offset);

    } else if (field->flags & VMS_VARRAY_UINT32) {

        n_elems = *(uint32_t *)(opaque+field->num_offset);

    } else if (field->flags & VMS_VARRAY_UINT16) {

        n_elems = *(uint16_t *)(opaque+field->num_offset);

    } else if (field->flags & VMS_VARRAY_UINT8) {

        n_elems = *(uint8_t *)(opaque+field->num_offset);

    }



    return n_elems;

}
