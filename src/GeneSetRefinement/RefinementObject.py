"""
Base class for objects that contain components of a refinement result.
"""

from typing import cast, Optional, Type, TypeVar


class RefinementObject:
	class RefinementNoneException(Exception):
		def __init__(
			self,
		) -> None:
			"""
			`obj_field_name` should be "ImplementingClass.field". 
			"""
			msg = ((
				f"Tried to access a field of {self.__class__.__name__} that "
				f"is None. If this is input data or a subset of it, "
				f"Refinement.save() may have been called with "
				f"`remove_inputs = True`"
			))

			super().__init__(msg)

	OBJ_T = TypeVar("OBJ_T")
	def assert_not_None(
		self,
		obj: Optional[OBJ_T]
	) -> OBJ_T:
		"""
		Either returns the object provided or raises a 
		`RefinementObject.RefinementNoneException`. This will both
		insure correctness and runtime and satisfy type checkers.
		"""
		if obj is None:
			raise self.RefinementNoneException
		
		return obj