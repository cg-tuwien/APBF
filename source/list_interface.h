namespace pbd
{
	template<class EditList>
	class list_interface
	{
	public:
		virtual void apply_edit(EditList& pEditList, list_interface* pEditSource) = 0;
	};
}
