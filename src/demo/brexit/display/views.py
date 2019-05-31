from django.shortcuts import render
from django.views import View

# Create your views here.


class Index(View):
    def get(self, request):
        context = {}
        return render(request, 'display/index.html', context)
