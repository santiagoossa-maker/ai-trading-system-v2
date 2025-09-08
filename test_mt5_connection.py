#!/usr/bin/env python3
"""
Test EXACTO de símbolos requeridos - AI Trading System V2
"""

import MetaTrader5 as mt5
import pandas as pd
from dotenv import load_dotenv
import os

load_dotenv()

def test_exact_symbols():
    """Buscar EXACTAMENTE los símbolos que necesitamos"""
    
    # Conectar MT5
    mt5.initialize()
    login = int(os.getenv('MT5_LOGIN'))
    password = os.getenv('MT5_PASSWORD')
    server = os.getenv('MT5_SERVER')
    mt5.login(login, password=password, server=server)
    
    print("🎯 BUSCANDO SÍMBOLOS EXACTOS REQUERIDOS")
    print("=" * 60)
    
    # MAPEO EXACTO que necesitamos
    required_mapping = {
        "1HZ75V": "Volatility 75 (1s) Index",
        "R_75": "Volatility 75 Index", 
        "R_100": "Volatility 100 Index",
        "1HZ100V": "Volatility 100 (1s) Index",
        "R_50": "Volatility 50 Index",
        "1HZ50V": "Volatility 50 (1s) Index",
        "R_25": "Volatility 25 Index",
        "1HZ25V": "Volatility 25 (1s) Index", 
        "R_10": "Volatility 10 Index",
        "1HZ10V": "Volatility 10 (1s) Index",
        "stpRNG": "Step Index",
        "stpRNG2": "Step Index 200",
        "stpRNG3": "Step Index 300", 
        "stpRNG4": "Step Index 400",
        "stpRNG5": "Step Index 500"
    }
    
    # Obtener todos los símbolos disponibles
    all_symbols = mt5.symbols_get()
    symbol_names = [s.name for s in all_symbols]
    symbol_descriptions = {s.name: s.description for s in all_symbols}
    
    print(f"Total símbolos en MT5: {len(all_symbols)}")
    print()
    
    # Verificar cada símbolo requerido
    found_mapping = {}
    missing_symbols = []
    
    for internal_name, expected_name in required_mapping.items():
        print(f"🔍 Buscando: {internal_name} → {expected_name}")
        
        # MÉTODO 1: Buscar por nombre exacto
        if expected_name in symbol_names:
            print(f"   ✅ ENCONTRADO (nombre exacto): {expected_name}")
            found_mapping[internal_name] = expected_name
            
        else:
            # MÉTODO 2: Buscar por palabras clave en descripción
            found = False
            
            # Extraer palabras clave
            if "volatility" in expected_name.lower():
                if "75" in expected_name and "(1s)" in expected_name:
                    keywords = ["volatility", "75", "1s", "second"]
                elif "75" in expected_name:
                    keywords = ["volatility", "75"]
                elif "100" in expected_name and "(1s)" in expected_name:
                    keywords = ["volatility", "100", "1s", "second"]
                elif "100" in expected_name:
                    keywords = ["volatility", "100"]
                elif "50" in expected_name and "(1s)" in expected_name:
                    keywords = ["volatility", "50", "1s", "second"]
                elif "50" in expected_name:
                    keywords = ["volatility", "50"]
                elif "25" in expected_name and "(1s)" in expected_name:
                    keywords = ["volatility", "25", "1s", "second"]
                elif "25" in expected_name:
                    keywords = ["volatility", "25"]
                elif "10" in expected_name and "(1s)" in expected_name:
                    keywords = ["volatility", "10", "1s", "second"]
                elif "10" in expected_name:
                    keywords = ["volatility", "10"]
                    
            elif "step index" in expected_name.lower():
                if "200" in expected_name:
                    keywords = ["step", "200"]
                elif "300" in expected_name:
                    keywords = ["step", "300"]
                elif "400" in expected_name:
                    keywords = ["step", "400"]
                elif "500" in expected_name:
                    keywords = ["step", "500"]
                else:
                    keywords = ["step", "index"]
            
            # Buscar por palabras clave
            for symbol_name in symbol_names:
                desc = symbol_descriptions.get(symbol_name, "").lower()
                name_lower = symbol_name.lower()
                
                if all(keyword in desc or keyword in name_lower for keyword in keywords):
                    print(f"   ✅ ENCONTRADO (por descripción): {symbol_name}")
                    print(f"      Descripción: {symbol_descriptions[symbol_name]}")
                    found_mapping[internal_name] = symbol_name
                    found = True
                    break
            
            if not found:
                print(f"   ❌ NO ENCONTRADO: {expected_name}")
                missing_symbols.append(internal_name)
        
        print()
    
    # PROBAR SÍMBOLOS ENCONTRADOS
    print("=" * 60)
    print("🧪 PROBANDO SÍMBOLOS ENCONTRADOS:")
    print("=" * 60)
    
    working_symbols = {}
    
    for internal_name, mt5_name in found_mapping.items():
        print(f"Probando {internal_name} → {mt5_name}")
        
        rates = mt5.copy_rates_from_pos(mt5_name, mt5.TIMEFRAME_M1, 0, 1)
        if rates is not None:
            price = rates[0]['close']
            print(f"   ✅ FUNCIONA - Precio: {price}")
            working_symbols[internal_name] = mt5_name
        else:
            error_code, error_desc = mt5.last_error()
            print(f"   ❌ Error: {error_desc}")
        print()
    
    # GUARDAR MAPEO CORRECTO
    print("=" * 60)
    print("💾 GUARDANDO MAPEO CORRECTO:")
    print("=" * 60)
    
    with open('symbol_mapping_exact.txt', 'w') as f:
        f.write("# MAPEO EXACTO DE SÍMBOLOS - AI TRADING SYSTEM V2\n")
        f.write(f"# Fecha: 2025-08-26 18:36:04\n")
        f.write(f"# Símbolos funcionando: {len(working_symbols)}/15\n\n")
        
        for internal_name, mt5_name in working_symbols.items():
            f.write(f'"{internal_name}": "{mt5_name}",\n')
            print(f'✅ {internal_name:10} → {mt5_name}')
    
    if missing_symbols:
        print(f"\n❌ Símbolos faltantes: {missing_symbols}")
    
    print(f"\n📁 Archivo guardado: symbol_mapping_exact.txt")
    
    mt5.shutdown()
    return len(working_symbols)

if __name__ == "__main__":
    found_count = test_exact_symbols()
    print(f"\n🎯 RESULTADO: {found_count}/15 símbolos encontrados y funcionando")