# Generated by Django 3.2.9 on 2022-04-09 14:05

from django.db import migrations, models


class Migration(migrations.Migration):

    initial = True

    dependencies = [
    ]

    operations = [
        migrations.CreateModel(
            name='Fencidata',
            fields=[
                ('id', models.AutoField(auto_created=True, primary_key=True, serialize=False, verbose_name='ID')),
                ('juzi', models.TextField(blank=True, verbose_name='句子')),
                ('username', models.CharField(blank=True, max_length=500, verbose_name='用户名')),
                ('result', models.TextField(blank=True, verbose_name='分词结果')),
                ('date', models.DateTimeField(auto_now=True, verbose_name='日期')),
            ],
            options={
                'verbose_name': '分词数据',
                'verbose_name_plural': '分词数据',
            },
        ),
        migrations.CreateModel(
            name='Keyworddata',
            fields=[
                ('id', models.AutoField(auto_created=True, primary_key=True, serialize=False, verbose_name='ID')),
                ('juzi', models.TextField(blank=True, verbose_name='句子')),
                ('username', models.CharField(blank=True, max_length=500, verbose_name='用户名')),
                ('result', models.TextField(blank=True, verbose_name='分词结果')),
                ('date', models.DateTimeField(auto_now=True, verbose_name='日期')),
            ],
            options={
                'verbose_name': '分词数据',
                'verbose_name_plural': '分词数据',
            },
        ),
        migrations.CreateModel(
            name='Lead1data',
            fields=[
                ('id', models.AutoField(auto_created=True, primary_key=True, serialize=False, verbose_name='ID')),
                ('juzi', models.TextField(blank=True, verbose_name='句子')),
                ('username', models.CharField(blank=True, max_length=500, verbose_name='用户名')),
                ('result', models.TextField(blank=True, verbose_name='抽取结果')),
                ('date', models.DateTimeField(auto_now=True, verbose_name='日期')),
            ],
            options={
                'verbose_name': '分词数据',
                'verbose_name_plural': '分词数据',
            },
        ),
        migrations.CreateModel(
            name='Simdata',
            fields=[
                ('id', models.AutoField(auto_created=True, primary_key=True, serialize=False, verbose_name='ID')),
                ('juzi', models.TextField(blank=True, verbose_name='句子')),
                ('username', models.CharField(blank=True, max_length=500, verbose_name='用户名')),
                ('result', models.TextField(blank=True, verbose_name='分词结果')),
                ('date', models.DateTimeField(auto_now=True, verbose_name='日期')),
            ],
            options={
                'verbose_name': '分词数据',
                'verbose_name_plural': '分词数据',
            },
        ),
        migrations.CreateModel(
            name='Textrankdata',
            fields=[
                ('id', models.AutoField(auto_created=True, primary_key=True, serialize=False, verbose_name='ID')),
                ('juzi', models.TextField(blank=True, verbose_name='句子')),
                ('username', models.CharField(blank=True, max_length=500, verbose_name='用户名')),
                ('result', models.TextField(blank=True, verbose_name='抽取结果')),
                ('date', models.DateTimeField(auto_now=True, verbose_name='日期')),
            ],
            options={
                'verbose_name': '分词数据',
                'verbose_name_plural': '分词数据',
            },
        ),
    ]
