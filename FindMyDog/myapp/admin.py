from django.contrib import admin

# Register your models here.
from django.contrib import admin
from .models import User, Dog, DogImage, LostDogReport, FoundDogReport, FoundDogImage, AdoptionParent,TrainingConfig


@admin.register(User)
class UserAdmin(admin.ModelAdmin):
    list_display = ("username", "email", "role", "location_lat", "location_lng")
    list_filter = ("role",)
    search_fields = ("username", "email", "phone", "line_id")


@admin.register(Dog)
class DogAdmin(admin.ModelAdmin):
    list_display = ("name", "age", "owner", "organization")
    search_fields = ("name", "owner__username", "organization__name")

@admin.register(DogImage)
class DogImageAdmin(admin.ModelAdmin):
    # เลือกฟิลด์ที่ต้องการให้โชว์ในหน้ารวม (List View)
    list_display = ("id", "dog", "has_embedding") 
    
    # ฟังก์ชันช่วยเช็คว่ามี Embedding หรือยัง (เพราะ BinaryField แสดงตรงๆ ไม่ได้)
    def has_embedding(self, obj):
        if obj.embedding_binary:
            return f"✅ Yes ({len(obj.embedding_binary)} bytes)"
        return "❌ No"
    
    has_embedding.short_description = 'Embedding Status'

    # ถ้าต้องการให้เห็นฟิลด์ในหน้าแก้ไข/ดูรายละเอียด
    readonly_fields = ("embedding_binary",)

admin.site.register(LostDogReport)
admin.site.register(FoundDogReport)
admin.site.register(FoundDogImage)
admin.site.register(AdoptionParent)
admin.site.register(TrainingConfig)