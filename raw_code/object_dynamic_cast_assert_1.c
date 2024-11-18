Object *object_dynamic_cast_assert(Object *obj, const char *typename)

{

    Object *inst;



    inst = object_dynamic_cast(obj, typename);



    if (!inst) {

        fprintf(stderr, "Object %p is not an instance of type %s\n",

                obj, typename);

        abort();

    }



    return inst;

}
