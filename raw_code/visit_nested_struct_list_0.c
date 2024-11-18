static void visit_nested_struct_list(Visitor *v, void **native, Error **errp)

{

    visit_type_UserDefNestedList(v, (UserDefNestedList **)native, NULL, errp);

}
